# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-turn warm-up reward (CUDA-Agent paper, sec 3.3 "Single-Turn Warm-up").

The model emits a full self-contained ``ModelNew`` (load_inline CUDA) in ONE
response. We drop it into a fresh workspace alongside the task's reference
``model.py`` and run the SAME authoritative harness as the agentic stage
(compile -> verify over 5 random inputs with ``F.*`` blocked -> profile eager /
torch.compile / custom), then assign the milestone reward {-1, 1, 2, 3}.

This warm-up builds CUDA capability and, crucially, produces reward variance
that pure agentic-from-cold does not — the paper's prerequisite for stable
agentic RL.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample

from .reward_cuda_agentic import _milestone
from .app.workspace import build_workspace, evaluate_final


logger = get_logger(__name__)

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str | None:
    blocks = _CODE_RE.findall(text or "")
    if not blocks:
        return None
    # The full solution is the longest fenced block (ignore tiny snippets).
    return max(blocks, key=len).strip() or None


def _evaluate(code: str, task_code: str) -> dict:
    root = tempfile.mkdtemp(prefix="cuda_st_")
    ws = os.path.join(root, "ws")
    try:
        build_workspace(ws, task_code)
        with open(os.path.join(ws, "model_new.py"), "w") as f:
            f.write(code)
        return evaluate_final(ws)
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def reward_func(args, sample: Sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    task = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_code = task.get("code")
    if not task_code:
        raise ValueError(f"single-turn task missing 'code' in metadata: {task!r}")

    code = _extract_code(sample.response)
    if code is None:
        return {"score": -1.0, "compiled": 0.0, "correct": 0.0,
                "speedup_vs_eager": 0.0, "speedup_vs_compile": 0.0, "has_code": 0.0}

    ev = await asyncio.to_thread(_evaluate, code, task_code)
    score = _milestone(ev)

    cuda_us, eager_us, compile_us = ev.get("cuda_us"), ev.get("eager_us"), ev.get("compile_us")
    if ev.get("error"):
        logger.warning("single-turn reward error (compiled=%s correct=%s): %s",
                       ev.get("compiled"), ev.get("correct"), str(ev["error"])[:400])

    return {
        "score": score,
        "has_code": 1.0,
        "compiled": 1.0 if ev.get("compiled") else 0.0,
        "correct": 1.0 if ev.get("correct") else 0.0,
        "speedup_vs_eager": float(eager_us / cuda_us) if (eager_us and cuda_us) else 0.0,
        "speedup_vs_compile": float(compile_us / cuda_us) if (compile_us and cuda_us) else 0.0,
    }
