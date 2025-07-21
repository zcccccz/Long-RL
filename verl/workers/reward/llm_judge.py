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

import torch
import ray
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM

from ...protocol import DataProto
from .config import RewardConfig
from .function import FunctionRewardManager


@ray.remote(num_gpus=1)
class LLMJudgeWorker:
    """A Ray remote worker that runs LLM for judging responses"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def judge_responses(self, prompts: List[str]) -> List[Dict[str, float]]:
        """Judge a batch of responses using the LLM"""
        with torch.no_grad():
            # Tokenize all prompts
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate responses
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                # Remove input tokens to get only the generated part
                generated = output[inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)
                responses.append(response)
            
            # Parse responses to extract scores
            scores = []
            for response in responses:
                score = self._parse_score(response)
                scores.append(score)
            
            return scores
    
    def _parse_score(self, response: str) -> Dict[str, float]:
        """Parse the LLM response to extract numerical scores"""
        # Simple parsing logic - you can customize this based on your prompt format
        response_lower = response.lower()
        
        # Look for score patterns like "score: 0.8" or "accuracy: 85%"
        import re
        
        # Try to find numerical scores
        score_match = re.search(r'score[:\s]*([0-9]*\.?[0-9]+)', response_lower)
        accuracy_match = re.search(r'accuracy[:\s]*([0-9]*\.?[0-9]+)', response_lower)
        correct_match = re.search(r'correct|right|accurate', response_lower)
        incorrect_match = re.search(r'incorrect|wrong|inaccurate', response_lower)
        
        overall_score = 0.0
        accuracy_score = 0.0
        
        if score_match:
            overall_score = float(score_match.group(1))
            # Normalize to [0,1] if it's a percentage
            if overall_score > 1.0:
                overall_score = overall_score / 100.0
        elif accuracy_match:
            accuracy_score = float(accuracy_match.group(1))
            if accuracy_score > 1.0:
                accuracy_score = accuracy_score / 100.0
            overall_score = accuracy_score
        elif correct_match and not incorrect_match:
            overall_score = 1.0
            accuracy_score = 1.0
        elif incorrect_match and not correct_match:
            overall_score = 0.0
            accuracy_score = 0.0
        else:
            # Default to 0.5 if unclear
            overall_score = 0.5
            accuracy_score = 0.5
        
        return {
            "overall": overall_score,
            "accuracy": accuracy_score,
            "format": 1.0  # Assume format is always correct for LLM judge
        }


class LLMJudgeRewardManager(FunctionRewardManager):
    """Reward manager that uses an LLM to judge response quality"""
    
    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        # Don't call super().__init__ since we're not using a function file
        self.config = config
        self.tokenizer = tokenizer
        self.diffusion = config.diffusion
        
        # Initialize LLM judge worker
        judge_model_path = getattr(config, 'judge_model_path', 'Qwen/Qwen2.5-7B-Instruct')
        self.judge_worker = LLMJudgeWorker.remote(judge_model_path)
        
        # Template for judging prompts
        self.judge_template = getattr(config, 'judge_template', self._default_judge_template())
    
    def _default_judge_template(self) -> str:
        return """Please evaluate the following response to determine if it correctly answers the question.

Question: {question}
Ground Truth: {ground_truth}
Response: {response}

Please analyze the response and provide:
1. Whether the response is correct (Yes/No)
2. A numerical accuracy score from 0.0 to 1.0
3. Brief reasoning

Format your response as:
Correct: [Yes/No]
Accuracy: [0.0-1.0]
Reasoning: [Your explanation]"""
    
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward using LLM judge for a batch of data"""
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        
        # Prepare prompts for LLM judge
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
        
        # Get judgments from LLM
        scores = ray.get(self.judge_worker.judge_responses.remote(judge_prompts))
        
        # Process scores
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)
        
        return reward_tensor, reward_metrics