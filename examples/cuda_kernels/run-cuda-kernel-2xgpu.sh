#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B CUDA-kernel-writing GRPO, 2xGPU colocate.
#
# The model writes a custom CUDA kernel; the reward sandbox compiles + runs it
# on GPU and scores correctness against a reference (see reward_cuda.py).
#
# Prereqs:
#   python examples/cuda_kernels/make_dataset.py --output-dir /root/cuda_kernels_data
#   hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
#
# Usage:
#   EXP_DIR=/root NUM_GPUS=2 bash examples/cuda_kernels/run-cuda-kernel-2xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n 2 | cut -d, -f1 | paste -sd ',')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/cuda-kernel}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-/root/cuda_kernels_data}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../checkpoints/qwen3-4B-cuda-kernel-agentic}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save ${SAVE_DIR}
   --load ${SAVE_DIR}
   --save-interval 20
)

PROMPT_SET=${DATA_DIR}/train.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   # Qwen3 is a thinking model; long <think> chains get truncated before the
   # final code block (-> empty reward). Disable thinking so it emits the
   # kernel directly and fits the response budget -> dense reward signal.
   --apply-chat-template-kwargs '{"enable_thinking": false}'
   --rollout-shuffle

   # Agentic multi-turn rollout: the model writes a kernel, the env compiles +
   # profiles it (nsys + cuobjdump) and feeds the report back so it can make the
   # kernel faster over up to `max_turns` attempts (see cuda_config.yaml).
   --custom-generate-function-path examples.deepeyes.rollout.generate
   --custom-config-path examples/cuda_kernels/cuda_config.yaml
   # Custom reward: score the BEST (correct + fastest) attempt across turns.
   --custom-rm-path examples.cuda_kernels.reward_cuda.reward_func
   --reward-key score
   --reward-max-concurrency 4

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 2048
   --rollout-max-context-len 9216
   --rollout-temperature 1.0

   --global-batch-size 16
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # Env ships flashinfer_python 0.6.3 (< SGLang-required 0.6.4); use triton
   # attention backend to avoid the flashinfer version assert on A100.
   --sglang-attention-backend triton
)

WANDB_ARGS=(
   --use-tensorboard
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-cuda-kernel-gpu2-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-cuda-kernel-gpu2-${now}.log
