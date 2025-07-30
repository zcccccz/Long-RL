# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
import ray
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer
from vllm import LLM, SamplingParams

from ...protocol import DataProto
from ...utils.torch_dtypes import PrecisionType
from .config import RewardConfig
from .function import FunctionRewardManager


@ray.remote
class VLLMJudgeWorker:
    """A Ray remote worker that uses vLLM for high-performance LLM judging"""
    
    def __init__(self, 
                 model_path: str, 
                 judge_gpu_count: int = 1,
                 gpu_memory_utilization: float = 0.8,
                 max_model_len: int = 4096,
                 trust_remote_code: bool = True):
        """
        Initialize vLLM judge worker
        
        Args:
            model_path: Path to the judge model
            judge_gpu_count: Number of GPUs to use for the judge model
            gpu_memory_utilization: GPU memory utilization ratio
            max_model_len: Maximum model length
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.judge_gpu_count = judge_gpu_count
        
        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM engine with tensor parallelism
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=judge_gpu_count,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype("bf16")),
            enforce_eager=True,  # For better performance with small batches
            disable_log_stats=True,
        )
        
        # Sampling parameters optimized for judging
        self.sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistent judgments
            max_tokens=256,   # Enough for judgment responses
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        print(f"VLLMJudgeWorker initialized with {judge_gpu_count} GPUs")
    
    def judge_responses(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Judge a batch of responses using vLLM"""
        if not prompts:
            return []
        
        # Generate responses using vLLM
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        # Parse outputs to get scores
        scores = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            score = self._parse_binary_score(generated_text)
            scores.append(score)
        
        return scores
    
    def _parse_binary_score(self, response: str) -> Dict[str, float]:
        """Parse the LLM response to extract binary scores (0 or 1 only)"""
        response_lower = response.lower().strip()
        
        # Initialize scores
        overall_score = 0.0
        accuracy_score = 0.0
        
        # Look for explicit binary indicators
        correct_patterns = [
            r'\bcorrect\s*:\s*yes\b',
            r'\baccuracy\s*:\s*1\.0\b',
            r'\baccuracy\s*:\s*100%?\b',
            r'\bcorrect\b(?!\s*:\s*no)',
            r'\bright\b',
            r'\baccurate\b',
            r'\byes\b(?=\s*$|\s*[.!])',  # Yes at end or before punctuation
        ]
        
        incorrect_patterns = [
            r'\bcorrect\s*:\s*no\b',
            r'\baccuracy\s*:\s*0\.0\b',
            r'\baccuracy\s*:\s*0%?\b',
            r'\bincorrect\b',
            r'\bwrong\b',
            r'\binaccurate\b',
            r'\bno\b(?=\s*$|\s*[.!])',  # No at end or before punctuation
        ]
        
        # Check for correct patterns
        correct_found = any(re.search(pattern, response_lower) for pattern in correct_patterns)
        incorrect_found = any(re.search(pattern, response_lower) for pattern in incorrect_patterns)
        
        # Binary decision logic
        if correct_found and not incorrect_found:
            overall_score = 1.0
            accuracy_score = 1.0
        elif incorrect_found and not correct_found:
            overall_score = 0.0
            accuracy_score = 0.0
        else:
            # If ambiguous, look for numerical scores and convert to binary
            score_match = re.search(r'(?:score|accuracy)[:\s]*([0-9]*\.?[0-9]+)', response_lower)
            if score_match:
                numeric_score = float(score_match.group(1))
                if numeric_score > 1.0:  # Percentage
                    numeric_score = numeric_score / 100.0
                # Binary threshold at 0.5
                overall_score = 1.0 if numeric_score >= 0.5 else 0.0
                accuracy_score = overall_score
            else:
                # Default to incorrect if unclear
                overall_score = 0.0
                accuracy_score = 0.0
        
        return {
            "overall": overall_score,
            "accuracy": accuracy_score,
            "format": 1.0  # Always assume format is correct for LLM judge
        }
    
    def get_gpu_count(self) -> int:
        """Return the number of GPUs used by this worker"""
        return self.judge_gpu_count


class VLLMJudgeRewardManager(FunctionRewardManager):
    """Reward manager that uses vLLM for high-performance LLM judging"""
    
    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        # Don't call super().__init__ since we're not using a function file
        self.config = config
        self.tokenizer = tokenizer
        self.diffusion = config.diffusion
        
        # Extract judge configuration
        judge_model_path = getattr(config, 'judge_model_path', 'Qwen/Qwen2.5-7B-Instruct')
        judge_gpu_count = getattr(config, 'judge_gpu_count', 1)
        judge_gpu_memory_utilization = getattr(config, 'judge_gpu_memory_utilization', 0.8)
        judge_max_model_len = getattr(config, 'judge_max_model_len', 4096)
        
        # Initialize vLLM judge worker with specified GPU count
        # Use scheduling strategy to avoid conflicts with RL workers' placement groups
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        import ray
        
        # Get available nodes and use the first one for judge model
        nodes = ray.nodes()
        available_nodes = [node for node in nodes if node['Alive'] and node['Resources'].get('GPU', 0) > 0]
        
        if available_nodes:
            # Use node affinity to place judge model on a specific node
            node_id = available_nodes[0]['NodeID']
            scheduling_strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=True)
            
            self.judge_worker = VLLMJudgeWorker.options(
                num_gpus=judge_gpu_count,
                scheduling_strategy=scheduling_strategy
            ).remote(
                model_path=judge_model_path,
                judge_gpu_count=judge_gpu_count,
                gpu_memory_utilization=judge_gpu_memory_utilization,
                max_model_len=judge_max_model_len,
                trust_remote_code=True
            )
        else:
            # Fallback to default scheduling if no nodes found
            self.judge_worker = VLLMJudgeWorker.options(
                num_gpus=judge_gpu_count
            ).remote(
                model_path=judge_model_path,
                judge_gpu_count=judge_gpu_count,
                gpu_memory_utilization=judge_gpu_memory_utilization,
                max_model_len=judge_max_model_len,
                trust_remote_code=True
            )
        
        # Store GPU count for resource management
        self.judge_gpu_count = judge_gpu_count
        
        # Template for judging prompts
        self.judge_template = getattr(config, 'judge_template', self._default_binary_judge_template())
        
        print(f"VLLMJudgeRewardManager initialized with {judge_gpu_count} GPUs for judge model")
    
    def _default_binary_judge_template(self) -> str:
        return """You are an expert evaluator. Please determine if the response correctly answers the question.

Question: {question}
Ground Truth: {ground_truth}
Response: {response}

Analyze the response and determine if it is correct. Consider:
1. Does it answer the question accurately?
2. Is it consistent with the ground truth?
3. Is the reasoning sound?

Respond with EXACTLY one of the following:
- "Correct: Yes" if the response is correct
- "Correct: No" if the response is incorrect

Your judgment:"""
    
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute binary reward (0 or 1) using vLLM judge for a batch of data"""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        # Prepare prompts for vLLM judge
        judge_prompts = []
        for i in range(len(data)):
            valid_response_ids = response_ids[i][:response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            # Extract question from the original prompt
            if "prompts" in data.batch:
                prompt_ids = data.batch["prompts"][i]
                question_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            else:
                question_str = "No question available"
            
            ground_truth = data.non_tensor_batch["ground_truth"][i]
            
            # Format the judge prompt
            judge_prompt = self.judge_template.format(
                question=question_str,
                ground_truth=ground_truth,
                response=response_str
            )
            judge_prompts.append(judge_prompt)
        
        # Get binary judgments from vLLM
        scores = ray.get(self.judge_worker.judge_responses.remote(judge_prompts))
        
        # Process binary scores (ensure they are 0 or 1)
        for i, score in enumerate(scores):
            # Ensure binary reward
            binary_reward = float(score["overall"])  # Should already be 0 or 1
            assert binary_reward in [0.0, 1.0], f"Reward must be binary, got {binary_reward}"
            
            reward_tensor[i, response_length[i] - 1] = binary_reward
            for key, value in score.items():
                reward_metrics[key].append(value)
        
        return reward_tensor, reward_metrics
    
    def get_judge_gpu_count(self) -> int:
        """Return the number of GPUs used by the judge model"""
        return self.judge_gpu_count


def calculate_available_gpus_for_rl(total_gpus: int, judge_gpu_count: int) -> int:
    """Calculate available GPUs for RL training after allocating GPUs for judge"""
    available_gpus = total_gpus - judge_gpu_count
    if available_gpus <= 0:
        raise ValueError(f"Not enough GPUs. Total: {total_gpus}, Judge needs: {judge_gpu_count}")
    
    # Ensure we have enough GPUs for meaningful RL training
    if available_gpus < 2:
        raise ValueError(f"Insufficient GPUs for RL training. Available: {available_gpus}, minimum required: 2")
    
    print(f"GPU allocation - Total: {total_gpus}, Judge: {judge_gpu_count}, RL: {available_gpus}")
    return available_gpus