# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Build the agentic CUDA-kernel dataset from CUDA-Agent-Ops-6K.

Each source row is a self-contained PyTorch module (``Model`` + ``get_inputs`` +
``get_init_inputs``). We turn it into an agentic-rollout row:

- ``prompt``   : chat messages (the SKILL workflow + the reference module). The
                 Relax chat API renders the tool schema from the agent's
                 ``tools=`` argument, so the model learns to drive
                 bash/read/write/edit over its workspace.
- ``metadata`` : ``code`` (the reference module written to ``model.py``), plus
                 ``ops`` / ``data_source`` / ``task_id`` for bookkeeping.

Usage:
    python examples/cuda_kernels_agentic/make_dataset.py \
        --src /root/cuda_agent_ops_6k/data.parquet \
        --output-dir /root/cuda_agentic_data [--limit N]
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


_SKILL = (Path(__file__).resolve().parent / "workdir_template" / "SKILL.md").read_text()

PROMPT_TEMPLATE = """{skill}

---

Your workspace already contains this reference module as `model.py` (READ-ONLY):

```python
{code}
```

Optimize it. Use the provided tools (`bash`, `read`, `write`, `edit`,
`multi_edit`, `glob`, `grep`) to write your CUDA kernels under `kernels/`, write
`model_new.py`, then `bash` `bash utils/compile.sh`, `python -m utils.verification`
and `python -m utils.profiling`. Iterate until verification passes and the CUDA
model beats `torch.compile`. Emit tool calls to act; stop when you are done."""


# Single-turn warm-up (CUDA-Agent paper, sec 3.3): the model emits the FULL
# solution in ONE response (no tools), to bootstrap CUDA capability and produce
# reward variance before the multi-turn agentic stage.
_LOAD_INLINE_EXAMPLE = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
__global__ void scale_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * 2.0f;
}
torch::Tensor scale_op(torch::Tensor x) {
    auto out = torch::empty_like(x);          // allocate output, never return a raw pointer
    int n = x.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""
cpp = "torch::Tensor scale_op(torch::Tensor x);"   # function DECLARATIONS only
_mod = load_inline(name="scale_ext", cpp_sources=cpp, cuda_sources=source,
                   functions=["scale_op"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return _mod.scale_op(x.contiguous())      # call the compiled module's function'''

SINGLE_TURN_TEMPLATE = """You are an expert CUDA engineer. Rewrite the following PyTorch module as `ModelNew` so it runs FASTER on GPU using a custom CUDA kernel, while producing identical outputs.

Reference module (`model.py`):
```python
{code}
```

Use `torch.utils.cpp_extension.load_inline` EXACTLY like this template (note the
argument names — there is no `sources=` argument):
```python
{example}
```

Write a SINGLE self-contained Python file defining `class ModelNew(nn.Module)` that:
- compiles its kernel with `load_inline(name=..., cpp_sources=<declarations>, cuda_sources=<kernel + wrapper>, functions=[...])`; the `<<<>>>` launch lives inside the C++ wrapper, never in Python;
- allocates outputs with `torch::empty_like(...)` / `torch::empty(...)` and returns that tensor (NEVER construct a tensor from a raw pointer);
- in `forward`, calls `_mod.<your_fn>(...)` on the compiled module (do NOT call the module object itself);
- has the SAME constructor signature and parameters/buffers as `Model`, so weights can be copied via `load_state_dict`;
- uses ONLY your custom CUDA op(s) plus basic tensor creation — NO `torch.nn.functional` calls, and do not just call the reference torch op.

Match `Model`'s output within atol=rtol=1e-2, then aim to beat `torch.compile` by >5%. Output EXACTLY one ```python code block with the full file, and nothing else."""


def build_prompt(code: str, mode: str) -> str:
    if mode == "single_turn":
        return SINGLE_TURN_TEMPLATE.format(code=code, example=_LOAD_INLINE_EXAMPLE)
    return PROMPT_TEMPLATE.format(skill=_SKILL, code=code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/root/cuda_agent_ops_6k/data.parquet")
    parser.add_argument("--output-dir", default="/root/cuda_agentic_data")
    parser.add_argument("--limit", type=int, default=None, help="Only emit the first N rows (for smoke tests).")
    parser.add_argument(
        "--mode", choices=["agentic", "single_turn"], default="agentic",
        help="agentic = multi-turn tool-use prompt; single_turn = one-shot warm-up prompt.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.src)
    if args.limit:
        df = df.iloc[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "train.jsonl")

    n = 0
    with open(out_path, "w") as f:
        for i, row in df.iterrows():
            code = row["code"]
            ops = list(row["ops"]) if row["ops"] is not None else []
            rec = {
                "prompt": [{"role": "user", "content": build_prompt(code, args.mode)}],
                "label": str(row.get("data_source", "")),
                "metadata": {
                    "code": code,
                    "ops": ops,
                    "data_source": str(row.get("data_source", "")),
                    "task_id": int(i),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
