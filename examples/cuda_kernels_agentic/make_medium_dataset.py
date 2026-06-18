# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Medium-difficulty curriculum for the single-turn warm-up.

Calibration finding on Qwen3-4B:
- the released fused 6K is too hard -> every sample fails (all reward -1);
- pure elementwise seed ops are too easy -> every sample is correct but none
  beats eager (all reward 1).
Both extremes give ZERO intra-group variance, so GRPO has no gradient.

This set targets the middle band that produces variance: reductions and
normalizations (softmax / layernorm / rmsnorm / row-sum / row-max / l2-norm).
A 4B writes a correct block-reduction kernel only some of the time (-> mix of
-1 and 1), and a correct *fused* reduction can beat eager's multi-pass
implementation (-> reward 2). A few fused-elementwise tasks are mixed in as an
easier floor. The reference ``Model`` may use ``torch.nn.functional``; only the
agent's ``ModelNew`` is forbidden from doing so (enforced in verification).
"""

from __future__ import annotations

import argparse
import json
import os

from make_dataset import build_prompt


_PLAIN = """import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {body}


def get_inputs():
    return [torch.randn({shape})]


def get_init_inputs():
    return []
"""

_AFFINE = """import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn({d}))
        self.bias = nn.Parameter(torch.randn({d}))

    def forward(self, x):
        return {body}


def get_inputs():
    return [torch.randn({shape})]


def get_init_inputs():
    return []
"""

_WEIGHT = """import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn({d}))

    def forward(self, x):
        return {body}


def get_inputs():
    return [torch.randn({shape})]


def get_init_inputs():
    return []
"""

# Reductions / normalizations over the last dim — the variance generators.
_PLAIN_OPS = [
    ("softmax", "torch.softmax(x, dim=-1)"),
    ("logsoftmax", "torch.log_softmax(x, dim=-1)"),
    ("rowsum", "x.sum(dim=-1)"),
    ("rowmax", "x.max(dim=-1).values"),
    ("rowmean", "x.mean(dim=-1)"),
    ("l2norm", "x / (x.norm(dim=-1, keepdim=True) + 1e-6)"),
    ("softmax_mul", "torch.softmax(x, dim=-1) * 2.0"),
]

_WEIGHT_OPS = [
    ("rmsnorm", "x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight"),
    ("weighted_sum", "(x * self.weight).sum(dim=-1)"),
]

_AFFINE_OPS = [
    ("layernorm", "F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)"),
    ("scale_bias_relu", "torch.relu(x * self.weight + self.bias)"),
    ("silu_bias", "x * torch.sigmoid(x) + self.bias"),
]

# (rows, reduction_dim) — dim large enough that a block reduction is non-trivial
# and a fused kernel has headroom over eager's multi-pass implementation.
_SHAPES = [(4096, 2048), (8192, 1024), (2048, 4096), (16384, 512)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/root/cuda_medium_data")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "train.jsonl")

    rows = []
    for rn, d in _SHAPES:
        shape = f"{rn}, {d}"
        for name, body in _PLAIN_OPS:
            rows.append((f"{name}_{rn}x{d}", _PLAIN.format(body=body, shape=shape), [name]))
        for name, body in _WEIGHT_OPS:
            rows.append((f"{name}_{rn}x{d}", _WEIGHT.format(body=body, shape=shape, d=d), [name]))
        for name, body in _AFFINE_OPS:
            rows.append((f"{name}_{rn}x{d}", _AFFINE.format(body=body, shape=shape, d=d), [name]))

    with open(out, "w") as f:
        for task_id, code, ops in rows:
            rec = {
                "prompt": [{"role": "user", "content": build_prompt(code, "single_turn")}],
                "label": "torch",
                "metadata": {"code": code, "ops": ops, "data_source": "medium", "task_id": task_id},
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} medium tasks to {out}")


if __name__ == "__main__":
    main()
