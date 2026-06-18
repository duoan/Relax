#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B agentic CUDA-kernel GRPO, 2xGPU colocate (CUDA-Agent style).
#
# The policy drives an OpenHands-style tool-use agent (bash/read/write/edit over
# a real per-session workspace). It writes CUDA kernels + model_new.py, compiles
# them, and runs verification + profiling (eager vs torch.compile vs custom). The
# milestone reward {-1,1,2,3} scores correctness + speedup vs both baselines.
#
# Prereqs:
#   python examples/cuda_kernels_agentic/make_dataset.py \
#       --src /root/cuda_agent_ops_6k/data.parquet --output-dir /root/cuda_agentic_data
#   hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507
#
# Usage:
#   EXP_DIR=/root NUM_GPUS=2 bash examples/cuda_kernels_agentic/run_cuda_agentic_2xgpu.sh

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

# Shrink the agentic chat-API/shard fleet so the 2 GPU placement groups can get
# their CPU on this 24-CPU box (default 16 replicas would eat all CPUs). Inject
# into the Ray job runtime env so the controller/serve replicas see it.
export RELAX_AGENTIC_SHARD_COUNT="${RELAX_AGENTIC_SHARD_COUNT:-4}"
RUNTIME_ENV_JSON=$(python3 -c "import json,os; d=json.loads(os.environ['RUNTIME_ENV_JSON']); d['env_vars']['RELAX_AGENTIC_SHARD_COUNT']=os.environ['RELAX_AGENTIC_SHARD_COUNT']; print(json.dumps(d))")
export RUNTIME_ENV_JSON

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/cuda-kernel-agentic}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-/root/cuda_agentic_data}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../checkpoints/qwen3-4B-instruct2507-cuda-agentic}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B-Instruct-2507/
   --ref-load ${MODEL_DIR}/Qwen3-4B-Instruct-2507/
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
   # Qwen3-4B-Instruct-2507 is non-thinking natively (no <think> blocks, no
   # enable_thinking kwarg needed) and much stronger at tool calling / coding.
   --rollout-shuffle

   # External OpenHands-style agent process (one per session).
   --use-agentic-rollout
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-timeout 3600
   # GPU-check bounds (RELAX_GPU_SLOTS / RELAX_CUDA_MEM_FRACTION) are set inside
   # run_agent_app.sh — the RELAX_ env prefix is reserved by --agent-env.

   # Milestone reward computed from the agent-recorded GPU measurements.
   --custom-rm-path examples.cuda_kernels_agentic.reward_cuda_agentic.reward_func
   --reward-key score
   --reward-max-concurrency 8

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 4096
   --rollout-max-response-len 2048
   --rollout-max-context-len 24576
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
   --tb-experiment-name qwen3-4b-cuda-agentic-gpu2-${now}
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
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-cuda-agentic-gpu2-${now}.log
