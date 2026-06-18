#!/usr/bin/env python3
"""Cross-process GPU slot semaphore.

Many agent sessions run concurrently and each needs the GPU for correctness /
performance checks. On a shared (colocated) GPU we bound how many checks run at
once with an ``flock``-based counting semaphore over a set of lock files, and we
cap the per-process memory fraction so a check never OOMs the rollout engine.

Honours env:
- ``RELAX_GPU_SLOTS``      number of concurrent GPU checks (default 2)
- ``RELAX_GPU_SLOT_DIR``   directory holding the lock files (default /tmp/relax_cuda_gpu_slots)
- ``RELAX_CUDA_MEM_FRACTION`` per-process GPU memory cap (default 0.15)
"""

import os
import time
from contextlib import contextmanager


@contextmanager
def gpu_slot(timeout: float = 1800.0):
    import fcntl

    slots = max(1, int(os.environ.get("RELAX_GPU_SLOTS", "2")))
    num_gpus = max(1, int(os.environ.get("RELAX_EVAL_NUM_GPUS", "1")))
    slot_dir = os.environ.get("RELAX_GPU_SLOT_DIR", "/tmp/relax_cuda_gpu_slots")
    os.makedirs(slot_dir, exist_ok=True)

    deadline = time.time() + timeout
    held = None
    held_i = 0
    while held is None:
        for i in range(slots):
            fh = open(os.path.join(slot_dir, f"slot_{i}.lock"), "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = fh
                held_i = i
                break
            except OSError:
                fh.close()
        if held is None:
            if time.time() > deadline:
                raise TimeoutError("timed out waiting for a GPU slot")
            time.sleep(0.5)
    # Pin this eval to a GPU (round-robin by slot index) BEFORE CUDA initializes,
    # so concurrent evals spread across the colocated cards instead of piling on
    # GPU 0. slot i -> GPU (i % num_gpus) guarantees <= ceil(slots/num_gpus) per GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(held_i % num_gpus)
    try:
        yield
    finally:
        import fcntl as _fcntl

        _fcntl.flock(held, _fcntl.LOCK_UN)
        held.close()


def apply_memory_cap() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    frac = float(os.environ.get("RELAX_CUDA_MEM_FRACTION", "0.15"))
    try:
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    except Exception:  # noqa: BLE001
        pass
