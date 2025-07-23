#!/bin/bash

# 示例脚本：展示如何使用不同的GPU配置运行vLLM判断模块

VIDEO_PATH="/path/to/your/video/data"

echo "=== vLLM Judge GPU Configuration Examples ==="

# 示例1: 8张GPU，判断模型使用1张GPU，RL使用7张GPU
echo "Example 1: 1 GPU for judge, 7 GPUs for RL"
# bash examples/qwen2_5_vl_3b_video_vllm_judge.sh $VIDEO_PATH 1 8

# 示例2: 8张GPU，判断模型使用2张GPU，RL使用6张GPU (推荐配置)
echo "Example 2: 2 GPUs for judge, 6 GPUs for RL (Recommended)"
# bash examples/qwen2_5_vl_3b_video_vllm_judge.sh $VIDEO_PATH 2 8

# 示例3: 16张GPU，判断模型使用4张GPU，RL使用12张GPU (大规模训练)
echo "Example 3: 4 GPUs for judge, 12 GPUs for RL (Large scale)"
# bash examples/qwen2_5_vl_3b_video_vllm_judge.sh $VIDEO_PATH 4 16

# 示例4: 4张GPU，判断模型使用1张GPU，RL使用3张GPU (小规模测试)
echo "Example 4: 1 GPU for judge, 3 GPUs for RL (Small scale testing)"
# bash examples/qwen2_5_vl_3b_video_vllm_judge.sh $VIDEO_PATH 1 4

echo ""
echo "Usage: bash examples/qwen2_5_vl_3b_video_vllm_judge.sh VIDEO_PATH [JUDGE_GPU_COUNT] [TOTAL_GPUS]"
echo ""
echo "GPU Allocation Guidelines:"
echo "- For 4-8 GPUs: Use 1-2 GPUs for judge"
echo "- For 8-16 GPUs: Use 2-4 GPUs for judge"
echo "- For 16+ GPUs: Use 4-8 GPUs for judge"
echo ""
echo "Factors to consider:"
echo "1. Judge model size (larger models need more GPUs)"
echo "2. Batch size for judgment (larger batches benefit from more GPUs)"
echo "3. RL model size and training requirements"
echo "4. Memory constraints"

# 取消注释下面的行来实际运行示例2
# bash examples/qwen2_5_vl_3b_video_vllm_judge.sh $VIDEO_PATH 2 8