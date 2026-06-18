# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Multi-turn interaction environment for agentic CUDA-kernel optimization.

Plugged into the generic deepeyes-style multi-turn rollout
(``examples.deepeyes.rollout.generate``) via ``rollout_interaction_env_path``.

Loop per sample:
    1. The model emits a CUDA kernel (one ```python block).
    2. ``step`` compiles + runs it in the sandbox and PROFILES it with the real
       tools available in this container — ``nsys`` (per-kernel GPU timing,
       launch counts) and ``cuobjdump`` (static registers / shared memory).
       ``ncu`` is unavailable (no GPU perf-counter permission), and that is
       stated in the feedback so the model does not ask for it.
    3. The profiler report is fed back as the next user turn so the model can
       make the kernel FASTER while staying correct.

Each turn's evaluation dict is appended to ``sample.metadata["_cuda_evals"]``;
the reward (``reward_cuda.reward_func``) scores the BEST attempt.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample

from .reward_cuda import _extract_code
from .sandbox import evaluate_kernel


logger = get_logger(__name__)

# Bound concurrent compile+profile jobs so many parallel rollouts do not
# oversubscribe CPU (nvcc) / GPU. Created lazily inside the running loop.
_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        n = int(os.environ.get("CUDA_KERNEL_SANDBOX_CONCURRENCY", "6"))
        _SEM = asyncio.Semaphore(max(1, n))
    return _SEM


def build_env(sample: Sample, args: Any) -> "CudaKernelEnv":
    return CudaKernelEnv(sample, args)


class CudaKernelEnv:
    def __init__(self, sample: Sample, args: Any) -> None:
        self.sample = sample
        self.args = args
        self.task = sample.metadata if isinstance(sample.metadata, dict) else {}
        self.max_turns = int(getattr(args, "max_turns", 3) or 3)
        self.timeout = float(getattr(args, "cuda_sandbox_timeout", 240.0))
        self.turn = 0

    def reset(self) -> None:
        self.turn = 0
        # Fresh eval log per rollout (metadata may be reused on resume).
        self.sample.metadata.setdefault("_cuda_evals", [])

    async def step(self, response_text: str):
        self.turn += 1
        attempt = self.turn

        code = _extract_code(response_text)
        if code is None:
            obs = (
                f"=== Feedback (attempt {attempt}/{self.max_turns}) ===\n"
                "No ```python code block found in your reply. Send the full "
                "solution inside exactly one ```python ... ``` block."
            )
            done = attempt >= self.max_turns
            return {"obs_str": obs}, done, {"has_code": 0.0}

        sem = _get_sem()
        async with sem:
            ev = await asyncio.to_thread(
                evaluate_kernel,
                code,
                self.task,
                timeout=self.timeout,
                do_timing=True,
                profile=True,
            )

        self.sample.metadata.setdefault("_cuda_evals", []).append(ev)

        obs = _format_report(ev, attempt, self.max_turns)
        done = attempt >= self.max_turns
        info = {
            "attempt": attempt,
            "compiled": 1.0 if ev.get("compiled") else 0.0,
            "correct": 1.0 if ev.get("correct") else 0.0,
            "speedup": float(ev["speedup"]) if ev.get("speedup") else 0.0,
        }
        return {"obs_str": obs}, done, info

    def close(self) -> None:
        pass

    def format_observation(self, observation: dict) -> dict:
        # Text-only model: keep content a plain string so the chat template
        # encodes it cleanly (no multimodal content list).
        return {"role": "user", "content": (observation or {}).get("obs_str", "")}


def _format_report(ev: dict, attempt: int, max_turns: int) -> str:
    lines = [f"=== Profiler feedback (attempt {attempt}/{max_turns}) ==="]

    if not ev.get("compiled"):
        err = (ev.get("error") or "").strip()
        lines.append("Compilation: FAILED")
        if err:
            lines.append(err[-700:])
        lines.append("Fix the error and resend the full solution in one ```python block.")
        return "\n".join(lines)

    lines.append("Compilation: OK")
    err_str = f" (max_abs_err={ev['max_abs_err']:.2e})" if ev.get("max_abs_err") is not None else ""

    if not ev.get("correct"):
        lines.append(f"Correctness: FAIL{err_str}")
        rt = (ev.get("error") or "").strip()
        if rt:
            lines.append(rt[-400:])
        lines.append("Your output does not match the reference. Fix correctness first, then optimize.")
        return "\n".join(lines)

    lines.append(f"Correctness: PASS{err_str}")

    lat, ref, sp = ev.get("latency_ms"), ev.get("ref_latency_ms"), ev.get("speedup")
    if lat and ref:
        rel = "faster than" if (sp and sp >= 1.0) else "slower than"
        lines.append(f"Performance (CUDA events): your kernel {lat:.3f} ms vs torch {ref:.3f} ms  ->  {sp:.2f}x ({rel} torch)")

    prof = ev.get("profile") or {}
    kernels = prof.get("kernels") or []
    if kernels:
        lines.append(f"nsys GPU kernel summary (your kernels only, {len(kernels)} distinct):")
        for k in kernels[:4]:
            lines.append(f"  - {k['name']}   avg {k['avg_us']:.1f} us   ({k['pct']:.0f}% of your GPU time)")
    res = prof.get("resources") or {}
    if res:
        regs = res.get("max_registers_per_thread")
        shm = res.get("max_shared_bytes_per_block")
        lines.append(f"cuobjdump static usage: {regs if regs is not None else '?'} regs/thread, "
                     f"{shm if shm is not None else '?'} B shared/block")

    hints = _hints(ev, prof)
    if hints:
        lines.append("Hints:")
        lines.extend(f"  - {h}" for h in hints)

    lines.append("Now rewrite the FULL solution to run FASTER while staying correct. "
                 "Reply with exactly one ```python block.")
    return "\n".join(lines)


def _hints(ev: dict, prof: dict) -> list[str]:
    hints: list[str] = []
    sp = ev.get("speedup") or 0.0
    res = prof.get("resources") or {}
    kernels = prof.get("kernels") or []

    if 0 < sp < 1.0:
        hints.append(f"You are {1.0 / sp:.1f}x slower than torch — improve memory coalescing "
                     "and use vectorized float4 loads/stores where the layout allows.")
    if len(kernels) > 1:
        hints.append(f"You launch {len(kernels)} distinct kernels; fuse them into one to cut launch "
                     "overhead and global-memory round-trips.")
    if res.get("max_shared_bytes_per_block", 0) == 0:
        hints.append("0 B shared memory used: row reductions and matmul should stage tiles in "
                     "__shared__ memory to avoid redundant global reads.")
    regs = res.get("max_registers_per_thread")
    if regs is not None and regs > 80:
        hints.append(f"{regs} registers/thread is high and can cap occupancy; reduce per-thread "
                     "work or split the kernel.")
    if prof.get("has_memcpy"):
        hints.append("Host<->device memcpy detected during execution; keep all tensors on-GPU.")
    return hints
