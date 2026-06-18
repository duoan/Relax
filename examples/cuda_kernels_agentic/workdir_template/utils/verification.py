#!/usr/bin/env python3
"""Correctness verification for the CUDA extension model.

Adapted from the CUDA-Agent reference harness. Builds the reference ``Model``
and the agent's ``ModelNew``, copies weights, and checks outputs over several
random inputs. The agent's forward runs inside ``block_torch_functional`` so it
cannot cheat by falling back to ``torch.nn.functional``.

Prints a single machine-readable line ``RELAX_VERIFY_JSON: {...}`` so the agent
loop / reward harness can parse the outcome without scraping free text.
"""

import json
import sys
import traceback
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from utils.gpu_slot import apply_memory_cap, gpu_slot


NUM_CHECKS = 5
ATOL = 1e-2
RTOL = 1e-2


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def check_equal(actual, expected):
    if type(actual) is not type(expected):
        raise AssertionError(f"type mismatch {type(actual)} != {type(expected)}")
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"len mismatch {len(actual)} != {len(expected)}")
        for x, y in zip(actual, expected):
            check_equal(x, y)
    elif isinstance(actual, dict):
        for key, val in expected.items():
            if key not in actual:
                raise AssertionError(f"missing key {key}")
            check_equal(actual[key], val)
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    elif isinstance(actual, (str, float, int)):
        if actual != expected:
            raise AssertionError(f"{actual} != {expected}")
    else:
        raise TypeError(f"unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    excludes = excludes or set()
    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(f"torch.nn.functional.{__name} is not allowed; use your custom CUDA op.")

            setattr(F, name, wrapper)
    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def initialize_models():
    from model import Model, get_init_inputs
    from model_new import ModelNew

    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]
    torch_model = Model(*init_inputs).eval().cuda()
    cuda_model = ModelNew(*init_inputs).eval().cuda()
    cuda_model.load_state_dict(torch_model.state_dict())
    return torch_model, cuda_model


def build_inputs():
    from model import get_inputs

    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())
    return torch_inputs, cuda_inputs


def run() -> dict:
    if not torch.cuda.is_available():
        return {"correct": False, "error": "CUDA not available"}
    apply_memory_cap()
    torch_model, cuda_model = initialize_models()
    max_abs_err = 0.0
    with torch.no_grad():
        for i in range(NUM_CHECKS):
            torch_inputs, cuda_inputs = build_inputs()
            torch_output = torch_model(*torch_inputs)
            with block_torch_functional():
                cuda_output = cuda_model(*cuda_inputs)
            if isinstance(torch_output, torch.Tensor) and isinstance(cuda_output, torch.Tensor):
                max_abs_err = max(max_abs_err, float((cuda_output.float() - torch_output.float()).abs().max().item()))
            check_equal(cuda_output, torch_output)
    torch.cuda.synchronize()
    return {"correct": True, "checks": NUM_CHECKS, "max_abs_err": max_abs_err}


def main() -> int:
    result = {"correct": False, "error": ""}
    try:
        with gpu_slot():
            result = run()
    except Exception:  # noqa: BLE001
        result = {"correct": False, "error": traceback.format_exc(limit=6)}
    print("RELAX_VERIFY_JSON: " + json.dumps(result))
    if result.get("correct"):
        print(f"[PASS] verify success (max_abs_err={result.get('max_abs_err')})")
        return 0
    print("[FAIL] verify failed:\n" + str(result.get("error", "")))
    return 1


if __name__ == "__main__":
    sys.exit(main())
