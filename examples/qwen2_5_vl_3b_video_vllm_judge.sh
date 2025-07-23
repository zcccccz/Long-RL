#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
JUDGE_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct  # replace it with your judge model path
VIDEO_PATH=$1

# Judge model configuration
JUDGE_GPU_COUNT=${2:-2}  # Default to 2 GPUs for judge, can be overridden
TOTAL_GPUS=${3:-8}       # Default to 8 total GPUs, can be overridden

num_video_frames=32
tokens_per_frame=60
max_num_batched_tokens=$((num_video_frames*tokens_per_frame+8192))

echo "=== GPU Allocation ==="
echo "Total GPUs: $TOTAL_GPUS"
echo "Judge GPUs: $JUDGE_GPU_COUNT"
echo "RL GPUs: $((TOTAL_GPUS - JUDGE_GPU_COUNT))"
echo "======================"

python3 -m verl.trainer.main \
    config=examples/config_vllm_judge.yaml \
    data.train_files=LongVideo-Reason/longvideo-reason@train \
    data.val_files=LongVideo-Reason/longvideo-reason@validation \
    data.video_dir=$VIDEO_PATH \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.padding_free=true \
    worker.actor.ulysses_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.num_video_frames=$num_video_frames \
    worker.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    worker.reward.reward_type=vllm_judge \
    worker.reward.judge_model_path=${JUDGE_MODEL_PATH} \
    worker.reward.judge_gpu_count=${JUDGE_GPU_COUNT} \
    worker.reward.judge_gpu_memory_utilization=0.8 \
    worker.reward.judge_max_model_len=4096 \
    trainer.experiment_name=qwen2_5_vl_3b_video_vllm_judge \
    trainer.n_gpus_per_node=${TOTAL_GPUS} \
    trainer.val_before_train=false \
    trainer.val_freq=-1