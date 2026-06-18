#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Managed-command entry for the CUDA kernel agent. Relax injects RELAX_* env
# vars (chat endpoint, session IO paths). We map them to the OpenAI SDK and set
# the GPU-check bounds, then run one agent session.

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Bound concurrent GPU correctness/perf checks on the shared (colocated) GPUs and
# cap their memory so they never OOM the rollout engine.
export RELAX_GPU_SLOTS="${RELAX_GPU_SLOTS:-2}"
export RELAX_CUDA_MEM_FRACTION="${RELAX_CUDA_MEM_FRACTION:-0.2}"
export RELAX_GPU_SLOT_DIR="${RELAX_GPU_SLOT_DIR:-/tmp/relax_cuda_gpu_slots}"

# Per-session build / compile caches so parallel sessions never clash.
export TORCHINDUCTOR_CACHE_DIR="${RELAX_SESSION_IO_DIR}/inductor_cache"
export MAX_JOBS="${MAX_JOBS:-4}"

python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
