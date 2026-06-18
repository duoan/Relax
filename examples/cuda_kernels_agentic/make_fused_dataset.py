# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fused-elementwise curriculum — the variance band for a 4B base.

Calibration on Qwen3-4B-Instruct-2507 showed a very narrow competence band:
  - single elementwise ops  -> always correct, no headroom  -> all reward 1
  - reductions / fused 6K    -> always fail                 -> all reward -1
Neither yields the intra-prompt reward variance GRPO needs.

This set chains several elementwise ops over a large memory-bound tensor. The
model can implement elementwise math correctly (so it mostly avoids -1), but the
reward then depends on whether it FUSES the chain into a single kernel: a fused
kernel does one read/write pass and beats eager's N-launch implementation by
>5% (reward 2), while a correct-but-unfused / per-op solution stays at reward 1,
and the occasional arithmetic/indexing slip gives -1. Run with
``RELAX_PROFILE_COMPILE=0`` so reward 2 (beats eager) is the top milestone.
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

# Each body is a chain of >=3 elementwise ops over the whole tensor -> several
# eager kernel launches, so a single fused custom kernel has real headroom.
_PLAIN_OPS = [
    ("relu_scale_bias", "torch.relu(x) * 2.0 + 1.0"),
    ("sigmoid_res", "torch.sigmoid(x) * 3.0 + x"),
    ("clamp_sq_scale", "torch.clamp(x, -3.0, 3.0) * torch.clamp(x, -3.0, 3.0) * 0.5"),
    ("gaussian", "torch.exp(-(x * x)) * 2.0"),
    ("mish", "x * torch.tanh(torch.log1p(torch.exp(x)))"),
    ("gelu_tanh", "0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))"),
    ("tanh_chain", "torch.tanh(x * 2.0) * 0.5 + 0.5"),
    ("softsign_scale", "(x / (1.0 + torch.abs(x))) * 4.0 - 1.0"),
]

_AFFINE_OPS = [
    ("affine_relu", "torch.relu(x * self.weight + self.bias) * 0.5 + 0.25"),
    ("silu_affine", "(x * torch.sigmoid(x)) * self.weight + self.bias"),
    ("tanh_affine", "torch.tanh(x) * self.weight + self.bias"),
]

# Big, memory-bound shapes so fusing N passes into 1 is a clear >5% win.
_SHAPES = [(8192, 8192), (4096, 16384), (16384, 4096), (8192, 4096)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/root/cuda_fused_data")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "train.jsonl")

    rows = []
    for rn, d in _SHAPES:
        shape = f"{rn}, {d}"
        for name, body in _PLAIN_OPS:
            rows.append((f"{name}_{rn}x{d}", _PLAIN.format(body=body, shape=shape), [name]))
        for name, body in _AFFINE_OPS:
            rows.append((f"{name}_{rn}x{d}", _AFFINE.format(body=body, shape=shape, d=d), [name]))

    with open(out, "w") as f:
        for task_id, code, ops in rows:
            rec = {
                "prompt": [{"role": "user", "content": build_prompt(code, "single_turn")}],
                "label": "torch",
                "metadata": {"code": code, "ops": ops, "data_source": "fused", "task_id": task_id},
            }
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(rows)} fused-elementwise tasks to {out}")


if __name__ == "__main__":
    main()
