# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Generate a small KernelBench-style dataset for CUDA-kernel-writing RL.

Each row asks the model to implement a single elementwise/simple operation as a
custom CUDA kernel exposed through PyTorch. The reward sandbox
(``examples.cuda_kernels.sandbox``) compiles the generated code, runs it on GPU
and checks correctness against the reference expression stored in ``metadata``.

Usage:
    python examples/cuda_kernels/make_dataset.py --output-dir /root/cuda_kernels_data
"""

import argparse
import json
import os


# Each task: a human-readable description plus a machine-checkable spec.
# ``reference`` is a torch expression evaluated by the trusted sandbox harness
# using the named inputs (and the ``torch`` module).
# Tasks are deliberately *optimizable*: a naive global-memory kernel is correct
# but much slower than a well-written one (vectorized loads, fused math, shared
# memory / warp reductions, tiled matmul). That gap is what the speedup-weighted
# reward rewards, so there is both correctness AND performance headroom to learn.
# The agent iterates with profiler feedback (nsys + cuobjdump) between turns.
_BIG = [1 << 20]  # 1M-element vectors: memory-bandwidth bound, perf-sensitive
_MAT = [512, 2048]  # 512 rows x 2048 cols: row reductions are the hard part
_SQ = [512, 512]  # square matmul: tiling / shared-memory headroom

TASKS = [
    # --- transcendental / fused elementwise (medium: perf varies with math) ---
    {
        "name": "gelu_tanh",
        "description": "elementwise GELU activation (tanh approximation)",
        "reference": "0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        "args": [("x", _BIG, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    {
        "name": "sigmoid",
        "description": "elementwise sigmoid activation 1/(1+exp(-x))",
        "reference": "torch.sigmoid(x)",
        "args": [("x", _BIG, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    {
        "name": "silu",
        "description": "elementwise SiLU/swish activation x*sigmoid(x)",
        "reference": "x * torch.sigmoid(x)",
        "args": [("x", _BIG, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    {
        "name": "fma_big",
        "description": "fused multiply-add a*b+c over large vectors",
        "reference": "a * b + c",
        "args": [("a", _BIG, "float32"), ("b", _BIG, "float32"), ("c", _BIG, "float32")],
        "rtol": 1e-3, "atol": 1e-3,
    },
    # --- row-wise reductions (hard: needs shared-memory / warp reduction) ------
    {
        "name": "softmax_rows",
        "description": "row-wise softmax over the last dimension of a 2-D tensor",
        "reference": "torch.softmax(x, dim=-1)",
        "args": [("x", _MAT, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    {
        "name": "layernorm_rows",
        "description": "row-wise LayerNorm over the last dim (eps=1e-5, no affine)",
        "reference": "(x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)",
        "args": [("x", _MAT, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    {
        "name": "rmsnorm_rows",
        "description": "row-wise RMSNorm over the last dim (eps=1e-6, no affine)",
        "reference": "x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)",
        "args": [("x", _MAT, "float32")],
        "rtol": 1e-2, "atol": 1e-2,
    },
    # --- matmul family (advanced: tiling + shared memory is the whole game) ----
    {
        "name": "matmul",
        "description": "square matrix multiply C = A @ B (A is MxK row-major, B is KxN row-major)",
        "reference": "a @ b",
        "args": [("a", _SQ, "float32"), ("b", _SQ, "float32")],
        "rtol": 2e-2, "atol": 2e-2,
    },
    {
        "name": "matmul_relu",
        "description": "fused matmul + ReLU: C = relu(A @ B)",
        "reference": "torch.relu(a @ b)",
        "args": [("a", _SQ, "float32"), ("b", _SQ, "float32")],
        "rtol": 2e-2, "atol": 2e-2,
    },
]

PROMPT_TEMPLATE = """You are an expert CUDA engineer. Implement the following operation as a custom CUDA kernel and expose it through PyTorch.

Operation: {description}.
Given the input tensor(s) ({arg_names}), the result must equal: {reference}

Follow this EXACT structure (the `load_inline` API and the CUDA launch happen inside `cuda_sources`, never with `<<<>>>` in Python):

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
__global__ void my_kernel(const float* x, float* out, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];  // TODO: replace with the real computation
}}
torch::Tensor run({cpp_params}) {{
    auto out = torch::empty_like({first_arg});
    int n = {first_arg}.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>({launch_args}, out.data_ptr<float>(), n);
    return out;
}}
'''
cpp_src = "torch::Tensor run({cpp_decl});"
mod = load_inline(name="kern", cpp_sources=cpp_src, cuda_sources=cuda_src, functions=["run"])

def solution({arg_names}):
    return mod.run({arg_names})
```

The template above only shows the `load_inline` mechanics for a 1-D elementwise
kernel. Adapt the kernel body, indexing and launch config to THIS operation —
e.g. row-wise reductions need one block per row with a shared-memory / warp
reduction; a matmul needs a 2-D grid and benefits from shared-memory tiling.

This is an ITERATIVE optimization task. After each attempt you will receive
profiler feedback (nsys kernel timings + launch counts, and cuobjdump
registers/shared-memory usage; ncu is unavailable in this sandbox). Use it to
make the kernel FASTER on the next turn — faster correct kernels score higher.

Requirements:
- Write the real `__global__` kernel yourself; do NOT just call a torch op.
- `solution` must accept the inputs in this exact order: {arg_names} (float32 CUDA tensors) and return the output tensor with the correct shape for the operation.
- Put ALL code in a single ```python code block, nothing else.
"""


def build_prompt(task: dict) -> str:
    names = [name for name, _, _ in task["args"]]
    arg_names = ", ".join(names)
    cpp_params = ", ".join(f"torch::Tensor {n}" for n in names)
    launch_args = ", ".join(f"{n}.data_ptr<float>()" for n in names)
    return PROMPT_TEMPLATE.format(
        description=task["description"],
        arg_names=arg_names,
        reference=task["reference"],
        cpp_params=cpp_params,
        cpp_decl=cpp_params,
        first_arg=names[0],
        launch_args=launch_args,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/cuda_kernels_data")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Duplicate the task list N times to enlarge the prompt pool.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.jsonl")

    rows = []
    for _ in range(args.repeat):
        for task in TASKS:
            rows.append(
                {
                    "prompt": [{"role": "user", "content": build_prompt(task)}],
                    "label": task["name"],
                    # The whole metadata dict IS the task spec; the reward reads it.
                    "metadata": {
                        "name": task["name"],
                        "entry_point": "solution",
                        "reference": task["reference"],
                        "args": task["args"],
                        "rtol": task.get("rtol", 1e-3),
                        "atol": task.get("atol", 1e-3),
                    },
                }
            )

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
