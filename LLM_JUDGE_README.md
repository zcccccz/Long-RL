# LLM Judge 模块使用指南

## 概述

本项目新增了高性能LLM判断模块，用于在强化学习过程中使用大语言模型来评估rollout结果的正确性。该模块支持两种实现方式：
1. **基础版本** (`llm_judge`): 使用transformers库，适合小规模测试
2. **高性能版本** (`vllm_judge`): 使用vLLM引擎，支持多GPU部署，适合生产环境

该模块会在强化学习启动前部署一个独立的LLM判断器，并自动管理GPU资源分配，确保与RL训练流程的资源隔离。

## 架构设计

### 核心组件

#### 基础版本 (llm_judge)
1. **LLMJudgeWorker**: Ray远程工作器，使用transformers库运行判断模型
2. **LLMJudgeRewardManager**: 基础奖励管理器

#### 高性能版本 (vllm_judge) - 推荐
1. **VLLMJudgeWorker**: Ray远程工作器，使用vLLM引擎高性能推理
2. **VLLMJudgeRewardManager**: 高性能奖励管理器，支持多GPU和批量推理
3. **资源管理系统**: 自动计算和分配GPU资源
4. **配置系统**: 扩展了原有的配置系统，支持多种判断模型配置

### 工作流程

1. **初始化阶段**: 
   - 系统启动时，会在指定的GPU上加载判断模型
   - 判断模型独立于主要的强化学习模型运行

2. **训练阶段**:
   - 每次rollout后，系统会将生成的回答发送给判断模块
   - 判断模块使用LLM对回答质量进行评估
   - 返回数值化的奖励分数用于强化学习更新

3. **评估流程**:
   - 提取原始问题、标准答案和模型回答
   - 构造判断提示词
   - 使用判断模型生成评估结果
   - 解析评估结果得到数值分数

## 使用方法

### 1. 配置文件设置

#### 高性能版本 (推荐)

使用 `examples/config_vllm_judge.yaml` 作为配置模板：

```yaml
worker:
  reward:
    reward_type: vllm_judge  # 使用vLLM高性能判断模块
    judge_model_path: Qwen/Qwen2.5-7B-Instruct  # 判断模型路径
    judge_gpu_count: 2  # 判断模型使用的GPU数量
    judge_gpu_memory_utilization: 0.8  # GPU内存利用率
    judge_max_model_len: 4096  # 最大模型长度
    judge_template: |  # 二值化判断提示词模板
      You are an expert AI evaluator...
      Important: You must respond with EXACTLY one of these:
      - "Correct: Yes" if correct
      - "Correct: No" if incorrect

trainer:
  n_gpus_per_node: 8  # 总GPU数量，系统会自动分配
```

#### 基础版本

使用 `examples/config_llm_judge.yaml` 作为配置模板：

```yaml
worker:
  reward:
    reward_type: llm_judge  # 基础判断模块
    judge_model_path: Qwen/Qwen2.5-7B-Instruct
    judge_template: |  # 自定义判断提示词模板
      Please carefully evaluate the following response...
```

### 2. 训练脚本

#### 高性能版本 (推荐)

```bash
# 基本使用：8张GPU，2张给判断模型，6张给RL
bash examples/qwen2_5_vl_3b_video_vllm_judge.sh /path/to/video/data

# 自定义GPU分配：总共16张GPU，4张给判断模型，12张给RL
bash examples/qwen2_5_vl_3b_video_vllm_judge.sh /path/to/video/data 4 16

# 小规模测试：总共4张GPU，1张给判断模型，3张给RL
bash examples/qwen2_5_vl_3b_video_vllm_judge.sh /path/to/video/data 1 4
```

#### 基础版本

```bash
bash examples/qwen2_5_vl_3b_video_llm_judge.sh /path/to/video/data
```

#### 手动运行

```bash
python3 -m verl.trainer.main \
    config=examples/config_vllm_judge.yaml \
    worker.reward.reward_type=vllm_judge \
    worker.reward.judge_model_path=Qwen/Qwen2.5-7B-Instruct \
    worker.reward.judge_gpu_count=2 \
    trainer.n_gpus_per_node=8 \
    # ... 其他参数
```

### 3. 自定义判断提示词

您可以通过配置文件中的 `judge_template` 字段自定义判断提示词：

```yaml
worker:
  reward:
    judge_template: |
      作为一个专业的AI评估员，请评估以下回答的质量：
      
      问题: {question}
      标准答案: {ground_truth}
      待评估回答: {response}
      
      请从以下角度评估：
      1. 回答是否正确
      2. 回答是否完整
      3. 推理是否合理
      
      请按以下格式输出：
      正确性: [是/否]
      准确度: [0.0-1.0]
      理由: [简短说明]
```

## 技术细节

### GPU资源分配

#### 自动资源管理
- **vLLM版本**: 支持多GPU tensor parallelism，可配置1-8张GPU
- **基础版本**: 使用单张GPU运行判断模型
- 系统自动计算可用于RL的GPU数量：`RL_GPUs = Total_GPUs - Judge_GPUs`
- 使用Ray的资源管理机制确保完全隔离

#### GPU分配策略
```
总GPU数量 = 4:  推荐 1张GPU给判断，3张给RL
总GPU数量 = 8:  推荐 2张GPU给判断，6张给RL  (默认配置)
总GPU数量 = 16: 推荐 4张GPU给判断，12张给RL
总GPU数量 = 32: 推荐 6-8张GPU给判断，24-26张给RL
```

### 性能优化

#### vLLM版本优势
- **高吞吐量**: vLLM引擎提供10-20x推理性能提升
- **批量推理**: 自动批处理多个判断请求
- **内存优化**: PagedAttention减少GPU内存碎片
- **Tensor并行**: 多GPU协同处理大模型推理
- **异步处理**: 与RL训练完全异步，不阻塞训练流程

#### 二值化奖励
- 强制输出0或1的二值奖励，避免中间值
- 优化的解析逻辑，支持多种判断格式
- 降级机制：模糊回答默认为0（错误）

### 分数解析

系统会自动解析判断模型的输出，支持多种格式：

1. 数值格式: "Score: 0.8", "Accuracy: 85%"
2. 文本格式: "Correct", "Incorrect", "Right", "Wrong"
3. 结构化格式: 按照提示词模板的格式输出

## 故障排除

### 常见问题

1. **GPU内存不足**
   - 调整 `judge_model_path` 使用更小的模型
   - 减少批处理大小

2. **判断结果不准确**
   - 优化 `judge_template` 提示词
   - 使用更强的判断模型

3. **性能问题**
   - 检查GPU资源分配
   - 调整批处理参数

### 调试建议

1. 启用详细日志查看判断过程
2. 使用小数据集进行测试
3. 检查判断模型的输出格式

## 扩展功能

### 支持多种判断模型

可以轻松切换不同的判断模型：

```yaml
worker:
  reward:
    judge_model_path: gpt-4  # 使用GPT-4作为判断模型
    # 或
    judge_model_path: claude-3-sonnet  # 使用Claude作为判断模型
```

### 自定义评估指标

可以扩展评估指标，不仅仅是准确性：

```python
def _parse_score(self, response: str) -> Dict[str, float]:
    # 解析多种评估维度
    return {
        "overall": overall_score,
        "accuracy": accuracy_score,
        "completeness": completeness_score,
        "reasoning": reasoning_score,
        "format": format_score
    }
```

## 最佳实践

1. **选择合适的判断模型**: 根据任务复杂度选择模型大小
2. **优化提示词**: 明确、具体的提示词能提高判断准确性
3. **监控性能**: 定期检查判断模块的性能指标
4. **资源管理**: 合理分配GPU资源避免冲突

## 贡献指南

欢迎提交改进建议和bug报告。主要改进方向：

1. 支持更多判断模型
2. 优化性能和资源使用
3. 改进评估准确性
4. 扩展评估维度

## 许可证

本模块遵循与主项目相同的Apache 2.0许可证。