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
- **总GPU资源**: 16个GPU由Ray统一管理
- **Judge模型**: 1个GPU (由Ray自动分配)
- **RL Workers**: 使用剩余可用的GPU资源进行训练
- **资源共享**: Judge模型和RL workers共享GPU资源池，Ray自动调度

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

### Ray自动资源管理

系统采用Ray的自动资源管理机制：

1. **资源池配置**: RL训练使用标准的资源池配置
   ```python
   resource_pool_spec = {
       "global_pool": [8, 8]  # 每个节点8个GPU
   }
   ```

2. **Judge模型部署**: 
   - Judge模型作为独立的Ray Actor部署
   - 使用 `num_gpus=1` 参数请求1个GPU
   - Ray自动从可用GPU中分配资源

3. **资源共享**: 
   - Judge模型和RL workers共享GPU资源池
   - Ray根据实际需求动态分配GPU
   - 避免资源冲突和死锁

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
  Judge model GPUs: 1 (deployed separately)
  RL training uses original resource pool: [8, 8]
  Note: Judge model and RL workers will share GPU resources automatically managed by Ray
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

这种情况下，Judge模型将使用2个GPU，RL workers将使用剩余的可用GPU资源。

### 不同节点配置

对于不同的节点配置，只需修改：

```yaml
trainer:
  nnodes: 4              # 4个节点
  n_gpus_per_node: 4     # 每个节点4个GPU
```

系统会自动管理资源：
- 总GPU: 16个
- Judge: 1个GPU (Ray自动分配)
- RL: 使用标准资源池 [4, 4, 4, 4]，与Judge模型共享GPU资源

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
- RL workers: Share GPU resources with judge model
- Total: 16 GPUs (2 nodes × 8 GPUs/node) managed by Ray
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

### 资源共享
系统采用Ray的自动资源管理，避免了手动GPU分配的复杂性：
- Judge模型和RL workers共享GPU资源池
- Ray根据实际需求动态分配GPU
- 避免了固定分配导致的资源浪费或冲突

这种方式更加灵活和高效。