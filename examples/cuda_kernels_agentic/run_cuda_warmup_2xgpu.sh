#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Stage 1 — Single-Turn Warm-up (CUDA-Agent paper, sec 3.3), 2xGPU colocate.
#
# The paper shows agentic RL launched from a cold base collapses (their trial
# died after 17 steps; ours produced all -1). The fix is a single-turn RL
# warm-up that first builds the model's intrinsic CUDA-generation ability and,
# crucially, produces reward variance. Here the policy emits a COMPLETE
# self-contained `ModelNew` (load_inline CUDA) in ONE response; the reward
# harness runs the SAME authoritative compile -> verify (5 inputs, F.* blocked)
# -> profile (eager / torch.compile / custom) pipeline and assigns the milestone
# reward {-1,1,2,3}. The resulting checkpoint then seeds the agentic stage.
#
# Prereqs:
#   python examples/cuda_kernels_agentic/make_dataset.py \
#       --src /root/cuda_agent_ops_6k/data.parquet \
#       --output-dir /root/cuda_warmup_data --mode single_turn
#   hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
#
# Usage:
#   EXP_DIR=/root NUM_GPUS=2 bash examples/cuda_kernels_agentic/run_cuda_warmup_2xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n 2 | cut -d, -f1 | paste -sd ',')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

# The reward spawns GPU subprocesses (verify/profile) on the colocated cards.
# Bound concurrency + per-process memory so they never OOM the rollout engine.
# RELAX_PROFILE_COMPILE=0 skips the slow torch.compile baseline during warm-up
# (reward then caps at 2 = beats eager), massively raising rollout throughput.
# 2 slots per card across both GPUs -> 4 concurrent evals (~2x throughput).
export RELAX_GPU_SLOTS="${RELAX_GPU_SLOTS:-4}"
export RELAX_EVAL_NUM_GPUS="${RELAX_EVAL_NUM_GPUS:-2}"
export RELAX_CUDA_MEM_FRACTION="${RELAX_CUDA_MEM_FRACTION:-0.12}"
export RELAX_PROFILE_COMPILE="${RELAX_PROFILE_COMPILE:-1}"
RUNTIME_ENV_JSON=$(python3 -c "import json,os; d=json.loads(os.environ['RUNTIME_ENV_JSON']); d['env_vars'].update({'RELAX_GPU_SLOTS':os.environ['RELAX_GPU_SLOTS'],'RELAX_EVAL_NUM_GPUS':os.environ['RELAX_EVAL_NUM_GPUS'],'RELAX_CUDA_MEM_FRACTION':os.environ['RELAX_CUDA_MEM_FRACTION'],'RELAX_PROFILE_COMPILE':os.environ['RELAX_PROFILE_COMPILE']}); print(json.dumps(d))")
export RUNTIME_ENV_JSON

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/cuda-kernel-warmup}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-/root/cuda_warmup_data}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../checkpoints/qwen3-4B-cuda-warmup}"
NUM_ROLLOUT="${NUM_ROLLOUT:=150}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B-Instruct-2507/
   --ref-load ${MODEL_DIR}/Qwen3-4B-Instruct-2507/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save ${SAVE_DIR}
   --load ${SAVE_DIR}
   --save-interval 10
)

PROMPT_SET=${DATA_DIR}/train.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   # prompt is a chat-message list; render it with the model's chat template.
   # Qwen3-4B-Instruct-2507 is non-thinking natively (no enable_thinking kwarg).
   --apply-chat-template

   # Single-turn: standard generation, full self-contained ModelNew in one shot.
   # Milestone reward built by compiling + verifying + profiling that response.
   --custom-rm-path examples.cuda_kernels_agentic.reward_single_turn.reward_func
   --reward-key score
   --reward-max-concurrency 6

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 4096
   --rollout-max-context-len 8192
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
   --max-tokens-per-gpu 12288
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
   --sglang-mem-fraction-static 0.6
   --sglang-attention-backend triton
)

WANDB_ARGS=(
   --use-tensorboard
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-cuda-warmup-gpu2-${now}
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
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-cuda-warmup-gpu2-${now}.log
