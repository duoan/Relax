# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stage-0 seed-operator curriculum (CUDA-Agent paper, Figure 1 / sec 3.1).

The paper's data pipeline starts from a repository of *seed operators*
(fundamental computational primitives crawled from PyTorch) and only later
fuses them into the hard multi-operator CUDA-Agent-Ops-6K set. The released 6K
is the fused/hard slice — far too hard for a 4B to solve one-shot, so a
single-turn warm-up on it yields zero reward variance (all -1) and no gradient.

This script synthesizes the missing easy slice: single-primitive KernelBench-
style tasks (elementwise unary ops, simple affine) that a 4B *can* implement
correctly in a custom CUDA kernel. Training the warm-up on these first builds
basic CUDA-generation skill and produces the reward variance the harder stages
need, exactly mirroring the paper's seed -> fusion curriculum.

Output format matches make_dataset.py (single_turn mode): each row carries the
single-turn prompt plus ``metadata.code`` (reference Model) for the reward.
"""

from __future__ import annotations

import argparse
import json
import os

from make_dataset import build_prompt


_MODEL_TMPL = """import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {body}


def get_inputs():
    return [{inp}]


def get_init_inputs():
    return []
"""

_MODEL_PARAM_TMPL = """import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn({pshape}))

    def forward(self, x):
        return {body}


def get_inputs():
    return [{inp}]


def get_init_inputs():
    return []
"""

# (name, forward body over `x`, input expression). All exactly implementable in
# a single elementwise CUDA kernel; inputs chosen to keep fp32 dynamic range
# safe so the atol/rtol=1e-2 check is meaningful.
_UNARY_OPS = [
    ("relu", "torch.relu(x)", "torch.randn({shape})"),
    ("abs", "torch.abs(x)", "torch.randn({shape})"),
    ("square", "x * x", "torch.randn({shape})"),
    ("neg", "-x", "torch.randn({shape})"),
    ("sigmoid", "torch.sigmoid(x)", "torch.randn({shape})"),
    ("tanh", "torch.tanh(x)", "torch.randn({shape})"),
    ("exp", "torch.exp(x)", "torch.randn({shape}) * 0.5"),
    ("sqrt", "torch.sqrt(x)", "torch.rand({shape}) + 0.1"),
    ("rsqrt", "torch.rsqrt(x)", "torch.rand({shape}) + 0.1"),
    ("sin", "torch.sin(x)", "torch.randn({shape})"),
    ("cos", "torch.cos(x)", "torch.randn({shape})"),
    ("log", "torch.log(x)", "torch.rand({shape}) + 0.5"),
    ("reciprocal", "torch.reciprocal(x)", "torch.rand({shape}) + 1.0"),
    ("leaky_relu", "torch.where(x > 0, x, 0.01 * x)", "torch.randn({shape})"),
    ("elu", "torch.where(x > 0, x, torch.exp(x) - 1)", "torch.randn({shape})"),
    ("clamp", "torch.clamp(x, -1.0, 1.0)", "torch.randn({shape}) * 2"),
    ("silu", "x * torch.sigmoid(x)", "torch.randn({shape})"),
    ("softplus", "torch.log1p(torch.exp(x))", "torch.randn({shape}) * 0.5"),
    ("gelu_tanh", "0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))", "torch.randn({shape})"),
    ("scale", "x * 2.5", "torch.randn({shape})"),
    ("add_const", "x + 3.0", "torch.randn({shape})"),
]

# Parameterized: a learned vector broadcast over the last dim (forces a correct
# load_state_dict-able ModelNew with matching parameter).
_PARAM_OPS = [
    ("bias_add", "x + self.weight", "torch.randn({shape})"),
    ("scale_vec", "x * self.weight", "torch.randn({shape})"),
]

_SHAPES = [
    (4096, 4096),
    (2048, 8192),
    (16, 1024, 1024),
    (8, 512, 2048),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/root/cuda_seed_data")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "train.jsonl")

    rows = []
    for name, body, inp in _UNARY_OPS:
        for shape in _SHAPES:
            shape_str = ", ".join(str(s) for s in shape)
            code = _MODEL_TMPL.format(body=body, inp=inp.format(shape=shape_str))
            rows.append((f"{name}_{'x'.join(map(str, shape))}", code, [name]))
    for name, body, inp in _PARAM_OPS:
        for shape in _SHAPES:
            shape_str = ", ".join(str(s) for s in shape)
            pshape = shape[-1]
            code = _MODEL_PARAM_TMPL.format(body=body, inp=inp.format(shape=shape_str), pshape=pshape)
            rows.append((f"{name}_{'x'.join(map(str, shape))}", code, [name]))

    with open(out, "w") as f:
        for task_id, code, ops in rows:
            rec = {
                "prompt": [{"role": "user", "content": build_prompt(code, "single_turn")}],
                "label": "torch",
                "metadata": {"code": code, "ops": ops, "data_source": "seed", "task_id": task_id},
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} seed tasks to {out}")


if __name__ == "__main__":
    main()
