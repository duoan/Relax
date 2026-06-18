# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Milestone reward for the agentic CUDA-kernel task (CUDA-Agent style).

The agent process already compiled, verified and profiled the final solution on
GPU and wrote the measurements into ``Sample.metadata`` (``compiled``,
``correct``, ``eager_us``, ``compile_us``, ``cuda_us``). This reward turns those
into the paper's robust discrete reward, which their ablation shows beats a raw
speedup reward:

    r = -1   if correctness fails
    r =  3   if faster than BOTH eager and torch.compile (>5%)
    r =  2   if faster than eager (>5%)
    r =  1   otherwise (correct but not faster)

When ``torch.compile`` is unavailable for an op (compile_us is null), the reward
is capped at 2 (we cannot certify it beats compile).
"""

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample


logger = get_logger(__name__)

SPEEDUP_THRESHOLD = 0.05  # must be >5% faster to count as a milestone


def _beats(baseline_us, cand_us) -> bool:
    if baseline_us is None or cand_us is None or cand_us <= 0:
        return False
    return (baseline_us - cand_us) / baseline_us > SPEEDUP_THRESHOLD


def _milestone(meta: dict) -> float:
    if not meta.get("correct"):
        return -1.0
    beats_eager = _beats(meta.get("eager_us"), meta.get("cuda_us"))
    beats_compile = _beats(meta.get("compile_us"), meta.get("cuda_us"))
    if beats_eager and beats_compile:
        return 3.0
    if beats_eager:
        return 2.0
    return 1.0


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}

    score = _milestone(meta)

    cuda_us = meta.get("cuda_us")
    eager_us = meta.get("eager_us")
    compile_us = meta.get("compile_us")
    speedup_vs_eager = (eager_us / cuda_us) if (eager_us and cuda_us) else 0.0
    speedup_vs_compile = (compile_us / cuda_us) if (compile_us and cuda_us) else 0.0

    if meta.get("error"):
        logger.debug("cuda-agentic reward task=%s error=%s", meta.get("stop_reason"), str(meta["error"])[:200])

    return {
        "score": score,
        "compiled": 1.0 if meta.get("compiled") else 0.0,
        "correct": 1.0 if meta.get("correct") else 0.0,
        "speedup_vs_eager": float(speedup_vs_eager),
        "speedup_vs_compile": float(speedup_vs_compile),
        "num_turns": float(meta.get("num_turns") or 0),
    }
