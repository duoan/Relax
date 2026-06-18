# CUDA Kernel Optimization Skill

You are a PyTorch + CUDA expert. Accelerate the reference model in `model.py` by
writing a custom CUDA C++ extension, then expose it as `ModelNew` in
`model_new.py`. Target the best possible speedup over `torch.compile` while
staying numerically correct (atol=rtol=1e-2 over 5 random inputs).

## Workspace layout

```
model.py                 # reference Model (READ-ONLY) — your target to match
model_new.py             # YOU WRITE: optimized model using custom ops
kernels/*.cu             # YOU WRITE: __global__ kernels + extern "C" launchers
kernels/*_binding.cpp    # YOU WRITE: pybind wrappers, REGISTER_BINDING(...)
binding.cpp              # READ-ONLY infrastructure
binding_registry.h       # READ-ONLY infrastructure (REGISTER_BINDING macro)
utils/                   # READ-ONLY: compile.sh, verification.py, profiling.py
```

## Hard rules (enforced)

- NO `torch::*` / `torch::nn::functional::*` inside `.cu` / `.cpp`.
- In `model_new.py`, NO `torch.nn.functional` in `forward` — only your custom
  `cuda_extension.*` ops and tensor creation. (Verification blocks `F.*`.)
- Do NOT modify `utils/`, `binding.cpp`, `binding_registry.h`, `model.py`
  (they are read-only; attempts are rejected).
- `ModelNew.__init__` MUST match `Model`'s parameters so `load_state_dict` works.

## Workflow (use the provided tools)

1. `read` `model.py` to understand the computation and tensor shapes.
2. `write` your kernel(s) in `kernels/<name>.cu` and binding in
   `kernels/<name>_binding.cpp` (register with `REGISTER_BINDING`).
3. `write` `model_new.py` with `class ModelNew(nn.Module)` calling
   `cuda_extension.<your_op>(...)`.
4. `bash`: `bash utils/compile.sh` to build `cuda_extension.so`.
5. `bash`: `python -m utils.verification` — must print `[PASS] verify success`.
6. `bash`: `python -m utils.profiling` — prints Eager / Compile / CUDA times.
7. Iterate: fuse ops, coalesce memory, use shared memory / float4 / warp
   reductions / TF32 to beat `torch.compile`. Keep the FINAL working version in
   `model_new.py` and `kernels/`.

## Optimization priorities

1. Algebraic simplification & operator fusion (biggest wins — avoid intermediate
   global-memory tensors, fold constants, exploit structure).
2. Memory coalescing, shared-memory tiling, vectorized (float4) loads.
3. Warp primitives, occupancy tuning, TF32 / Tensor Cores for GEMM/conv
   (cuBLAS allowed for GEMM, cuDNN for conv only).

When you are done, make sure `model_new.py` + `kernels/` contain ONLY the final
optimized version and that verification passes.
