# 多节点GPU分配指南

本文档说明如何配置多节点GPU环境，实现answer judge LLM和RL workers的合理GPU分配。

## 概述

在多节点训练环境中，我们需要：
1. **Answer Judge LLM**: 使用1个GPU部署vLLM推理服务
2. **RL Workers**: 使用剩余的GPU进行强化学习训练

## 2节点16GPU配置示例

### 硬件配置
- **节点数量**: 2个节点
- **每节点GPU**: 8个GPU
- **总GPU数量**: 16个GPU

### GPU分配策略
- **Judge模型**: 1个GPU (独立分配)
- **RL Workers**: 15个GPU，分布为 [8, 7] 跨2个节点
- **总计**: 16个GPU (2节点 × 8GPU/节点)

### 配置文件

使用 `examples/config_vllm_judge_2nodes.yaml` 作为配置模板：

```yaml
trainer:
  nnodes: 2                    # 使用2个节点
  n_gpus_per_node: 8          # 每个节点8个GPU

worker:
  reward:
    reward_type: vllm_judge    # 使用vLLM高性能判断模块
    judge_gpu_count: 1         # Judge模型使用1个GPU
    judge_model_path: Qwen/Qwen2.5-7B-Instruct
    judge_gpu_memory_utilization: 0.8
    judge_max_model_len: 4096
```

## GPU分配逻辑

### 自动GPU分配机制

系统会自动处理GPU分配：

1. **总GPU计算**: `total_gpus = nnodes × n_gpus_per_node = 2 × 8 = 16`
2. **RL可用GPU**: `available_gpus_for_rl = total_gpus - judge_gpu_count = 16 - 1 = 15`
3. **节点分布计算**:
   - `base_gpus_per_node = 15 // 2 = 7`
   - `extra_gpus = 15 % 2 = 1`
   - 第一个节点: `7 + 1 = 8` 个GPU
   - 第二个节点: `7 + 0 = 7` 个GPU

### 资源池规格

系统会自动生成以下资源池配置：

```python
resource_pool_spec = {
    "global_pool": [8, 7]  # 节点0: 8个GPU, 节点1: 7个GPU
}
```

## 运行示例

### 启动训练

```bash
# 使用2节点16GPU配置
python -m verl.trainer.main --config examples/config_vllm_judge_2nodes.yaml
```

### 预期输出

训练启动时会显示GPU分配信息：

```
GPU allocation summary:
  Total GPUs: 16 (2 nodes × 8 GPUs/node)
  Judge model GPUs: 1
  RL workers GPUs: 15 distributed as [8, 7]
  Resource pool spec: {'global_pool': [8, 7]}
VLLMJudgeRewardManager initialized with 1 GPUs for judge model
```

## 其他配置选项

### 自定义Judge GPU数量

如果需要为Judge模型分配更多GPU（例如使用更大的模型），可以调整：

```yaml
worker:
  reward:
    judge_gpu_count: 2  # 使用2个GPU for judge
```

这种情况下，RL workers将获得14个GPU，分布为 [7, 7]。

### 不同节点配置

对于不同的节点配置，只需修改：

```yaml
trainer:
  nnodes: 4              # 4个节点
  n_gpus_per_node: 4     # 每个节点4个GPU
```

系统会自动计算并分配：
- 总GPU: 16个
- Judge: 1个GPU  
- RL: 15个GPU，分布为 [4, 4, 4, 3]

## 验证配置

使用提供的测试脚本验证配置：

```bash
python3 simple_test_gpu_allocation.py
```

预期输出：
```
🎉 All tests passed!

Expected behavior:
- Judge model: 1 GPU (automatically allocated by Ray)
- RL workers: 15 GPUs distributed as [8, 7] across 2 nodes
- Total: 16 GPUs (2 nodes × 8 GPUs/node)
```

## 注意事项

1. **Ray集群**: 确保Ray集群已正确配置，所有节点都可访问
2. **GPU可见性**: 确保所有GPU对Ray可见
3. **模型路径**: 确保所有节点都能访问模型文件
4. **网络连接**: 确保节点间网络连接正常，支持分布式训练

## 故障排除

### GPU不足错误
如果遇到 "Not enough GPUs" 错误，检查：
- Ray集群中可用GPU数量
- 配置文件中的GPU设置
- 是否有其他任务占用GPU

### 不均匀分配
系统会自动处理不均匀的GPU分配。例如：
- 15个GPU分配到2个节点 → [8, 7]
- 14个GPU分配到3个节点 → [5, 5, 4]

这是正常行为，不会影响训练效果。分布式训练的world_size将等于RL workers的总GPU数量。