# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Custom reward for the CUDA-kernel-writing task.

Wired in via ``--custom-rm-path examples.cuda_kernels.reward_cuda.reward_func``
and ``--reward-key score``. The reward extracts a code block from the model
response, compiles + runs it in an isolated sandbox (see ``sandbox.py``) and
scores it correctness- AND performance-aware:

    score = 0.1 * has_code
          + 0.2 * compiled
          + 0.4 * correct
          + 0.3 * perf            # perf = min(speedup_vs_torch, 1.0), correct only

So among correct kernels, faster ones (better memory access, fused math, warp
reductions) score higher — that is the "not just correct, but fast" signal.
``speedup = torch_reference_time / kernel_time`` is measured back-to-back in the
same process so shared-GPU contention cancels out of the ratio.

Returning a dict lets the trainer log per-component metrics; GRPO uses the
``score`` sub-key (selected by ``--reward-key score``).
"""

import asyncio
import re

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample

from .sandbox import evaluate_kernel


logger = get_logger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python|cpp|c\+\+|cuda|c)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_code(response: str) -> str | None:
    """Return the last fenced code block, or None if there is none."""
    blocks = _CODE_BLOCK_RE.findall(response or "")
    if not blocks:
        return None
    # Models usually emit the final answer in the last fenced block.
    return blocks[-1].strip()


def _score_from_eval(ev: dict, perf_weight: float) -> float:
    score = 0.1  # has_code (reached only when a block exists)
    if ev.get("compiled"):
        score += 0.2
    if ev.get("correct"):
        score += 0.4
        # perf in [0, 1]: 1.0 == reaching torch parity; correct-but-slow naive
        # kernels land well below, giving a continuous "make it faster" gradient.
        speedup = ev.get("speedup") or 0.0
        perf = max(0.0, min(speedup, 1.0))
        score += perf_weight * perf
    return score


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    task = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "args" not in task or "reference" not in task:
        raise ValueError(f"CUDA task spec missing in sample.metadata: {task!r}")

    perf_weight = float(getattr(args, "cuda_perf_weight", 0.3))
    timeout = float(getattr(args, "cuda_sandbox_timeout", 240.0))

    # Agentic path: the interaction env (env_cuda) already compiled + profiled
    # every turn and stashed the per-turn evaluations. Score the BEST attempt so
    # the reward credits the optimization the agent achieved across turns.
    evals = sample.metadata.get("_cuda_evals") if isinstance(sample.metadata, dict) else None

    if not evals:
        # Single-turn fallback: no env ran, score the final response directly.
        code = _extract_code(sample.response)
        if code is None:
            return {
                "score": 0.0, "has_code": 0.0, "compiled": 0.0,
                "correct": 0.0, "speedup": 0.0, "turns": 0, "error": "no code block",
            }
        ev = await asyncio.to_thread(evaluate_kernel, code, task, timeout=timeout, do_timing=True)
        evals = [ev]

    scored = [(_score_from_eval(ev, perf_weight), ev) for ev in evals]
    best_score, best = max(scored, key=lambda s: s[0])

    if best.get("error"):
        logger.debug("cuda reward task=%s error=%s", task.get("name"), best["error"][:200])

    return {
        "score": best_score,
        "has_code": 1.0,
        "compiled": 1.0 if best.get("compiled") else 0.0,
        "correct": 1.0 if best.get("correct") else 0.0,
        "speedup": float(best["speedup"]) if best.get("speedup") else 0.0,
        "turns": len(evals),
        "error": (best.get("error") or "")[:200],
    }
