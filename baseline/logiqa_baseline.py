#!/usr/bin/env python3
"""
LogiQA Baseline Evaluation for Llama-3.2-3B
Supports simple inference, FSDP, and DeepSpeed optimization
"""

import argparse
import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed
import logging
import sys
import os

# Add current directory to path for logiqa_data import
sys.path.append(os.path.dirname(__file__))
from logiqa_data import load_logiqa_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogiQAEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.results = []
        
    def load_model(self):
        """Load Llama-3.2-3B model with specified optimization"""
        logger.info(f"Loading {self.args.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.args.use_deepspeed:
            # DeepSpeed initialization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model = deepspeed.initialize(
                model=self.model,
                config_params=self.args.deepspeed_config
            )[0]
            
        elif self.args.use_fsdp:
            # FSDP initialization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            self.model = FSDP(self.model)
            
        else:
            # Simple inference - optimized for 4GB GPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "3.5GB"}  # Reserve memory for operations
            )
            
        logger.info("Model loaded successfully")
        
    def load_logiqa_dataset(self):
        """Load LogiQA dataset using existing utils"""
        logger.info("Loading LogiQA dataset...")
        dataset = load_logiqa_dataset()
        logger.info(f"Loaded {len(dataset)} LogiQA examples")
        return dataset
            
    def format_prompt(self, example: Dict) -> str:
        """Format LogiQA example as prompt"""
        context = example.get('context', '')
        question = example.get('question', '')
        options = example.get('options', [])
        
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nOptions:\n"
        for i, option in enumerate(options):
            prompt += f"{chr(65+i)}. {option}\n"
        prompt += "\nAnswer:"
        
        return prompt
        
    def generate_response(self, prompt: str) -> str:
        """Generate model response"""
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.args.max_length
        )
        
        if not self.args.use_deepspeed and not self.args.use_fsdp:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                do_sample=self.args.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return response
        
    def extract_answer(self, response: str) -> str:
        """Extract answer choice from response"""
        response = response.upper().strip()
        
        # Look for single letter answers
        for choice in ['A', 'B', 'C', 'D', 'E']:
            if choice in response:
                return choice
                
        # Fallback to first character if it's a valid choice
        if response and response[0] in ['A', 'B', 'C', 'D', 'E']:
            return response[0]
            
        return "UNKNOWN"
        
    def evaluate_dataset(self):
        """Evaluate model on LogiQA dataset"""
        dataset = self.load_logiqa_dataset()
        
        if self.args.max_samples:
            dataset = dataset.select(range(min(self.args.max_samples, len(dataset))))
            
        correct = 0
        total = 0
        
        logger.info(f"Evaluating on {len(dataset)} examples...")
        start_time = time.time()
        
        for i, example in enumerate(dataset):
            if i % 50 == 0:
                logger.info(f"Processing example {i+1}/{len(dataset)}")
                
            prompt = self.format_prompt(example)
            response = self.generate_response(prompt)
            predicted_answer = self.extract_answer(response)
            
            # Get correct answer
            correct_answer = example.get('answer', example.get('label', ''))
            if isinstance(correct_answer, int):
                correct_answer = chr(65 + correct_answer)  # Convert 0,1,2,3 to A,B,C,D
                
            is_correct = predicted_answer == correct_answer.upper()
            if is_correct:
                correct += 1
            total += 1
            
            # Store result
            result = {
                'example_id': i,
                'context': example.get('context', ''),
                'question': example.get('question', ''),
                'options': example.get('options', []),
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct
            }
            self.results.append(result)
            
        end_time = time.time()
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy
        
    def save_results(self):
        """Save evaluation results"""
        output_file = Path(self.args.output_dir) / f"logiqa_results_{self.args.model_name.split('/')[-1]}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate final metrics
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        accuracy = correct / total if total > 0 else 0
        
        summary = {
            'model_name': self.args.model_name,
            'total_examples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'optimization': {
                'use_fsdp': self.args.use_fsdp,
                'use_deepspeed': self.args.use_deepspeed,
                'simple_inference': not (self.args.use_fsdp or self.args.use_deepspeed)
            },
            'generation_params': {
                'max_new_tokens': self.args.max_new_tokens,
                'temperature': self.args.temperature,
                'do_sample': self.args.do_sample
            },
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        
        # Also save a summary file
        summary_file = Path(self.args.output_dir) / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"LogiQA Baseline Results\n")
            f.write(f"Model: {self.args.model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")
            f.write(f"Optimization: {'FSDP' if self.args.use_fsdp else 'DeepSpeed' if self.args.use_deepspeed else 'Simple'}\n")
            
        logger.info(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="LogiQA Baseline Evaluation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B",
                       help="Model name or path")
    
    # Optimization arguments
    parser.add_argument("--use_fsdp", action="store_true",
                       help="Use FSDP for parallelization")
    parser.add_argument("--use_deepspeed", action="store_true", 
                       help="Use DeepSpeed for optimization")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json",
                       help="DeepSpeed configuration file")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="",
                       help="Path to local LogiQA dataset (fallback)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling for generation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./baseline_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_fsdp and args.use_deepspeed:
        raise ValueError("Cannot use both FSDP and DeepSpeed simultaneously")
        
    # Initialize evaluator
    evaluator = LogiQAEvaluator(args)
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    accuracy = evaluator.evaluate_dataset()
    
    # Save results
    evaluator.save_results()
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()