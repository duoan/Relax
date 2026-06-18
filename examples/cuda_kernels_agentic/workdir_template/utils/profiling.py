#!/usr/bin/env python3
"""Performance profiling: time eager vs torch.compile vs the custom CUDA model.

Adapted from the CUDA-Agent reference harness. Uses CUDA events (warmup +
repeated measurement) rather than CUPTI/torch.profiler so it works in containers
without GPU performance-counter permissions. Times are averaged over ``--iters``
and reported in microseconds.

Prints ``RELAX_PROFILE_JSON: {...}`` with ``eager_us``, ``compile_us``,
``cuda_us`` for the agent loop / reward harness to parse.
"""

import argparse
import json
import os
import sys
import traceback

import torch

from utils.gpu_slot import apply_memory_cap, gpu_slot


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def initialize():
    from model import Model, get_init_inputs, get_inputs
    from model_new import ModelNew

    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]
    torch_model = Model(*init_inputs).eval().cuda()
    cuda_model = ModelNew(*init_inputs).eval().cuda()
    cuda_model.load_state_dict(torch_model.state_dict())

    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())
    return torch_model, cuda_model, torch_inputs, cuda_inputs


def bench(model, inputs, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(*inputs)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # ms -> us


def run(iters: int) -> dict:
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    apply_memory_cap()
    torch_model, cuda_model, torch_inputs, cuda_inputs = initialize()
    warmup = 5

    cuda_us = bench(cuda_model, cuda_inputs, warmup, iters)
    eager_us = bench(torch_model, torch_inputs, warmup, iters)

    # The torch.compile (Inductor) baseline is the most expensive part of eval
    # (cold compile ~30-60s/sample). During the single-turn warm-up it can be
    # skipped (RELAX_PROFILE_COMPILE=0) to massively raise rollout throughput;
    # the milestone reward then caps at 2 (beats eager) since reward 3 needs it.
    compile_us = None
    if os.environ.get("RELAX_PROFILE_COMPILE", "1") != "0":
        try:
            compiled = torch.compile(torch_model)
            compile_us = bench(compiled, torch_inputs, warmup, iters)
        except Exception:  # noqa: BLE001
            compile_us = None  # torch.compile may fail for some ops; treat as unavailable

    return {"eager_us": eager_us, "compile_us": compile_us, "cuda_us": cuda_us}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    result = {"error": ""}
    try:
        with gpu_slot():
            result = run(args.iters)
    except Exception:  # noqa: BLE001
        result = {"error": traceback.format_exc(limit=6)}
    print("RELAX_PROFILE_JSON: " + json.dumps(result))
    if "error" in result and result["error"]:
        print("[FAIL] profiling failed:\n" + str(result["error"]))
        return 1
    e, c, u = result.get("eager_us"), result.get("compile_us"), result.get("cuda_us")
    cs = f"{c:.1f}us" if c is not None else "n/a"
    print(f"[DONE] Eager: {e:.1f}us, Compile: {cs}, CUDA: {u:.1f}us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
