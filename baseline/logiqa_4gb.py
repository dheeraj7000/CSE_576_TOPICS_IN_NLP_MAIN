#!/usr/bin/env python3
"""
Memory-optimized LogiQA evaluation for 4GB GPU
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logiqa_data import load_logiqa_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogiQA4GB:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model with 4GB GPU optimization"""
        logger.info("Loading model for 4GB GPU...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Ultra memory-efficient loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "3.2GB"},  # Conservative limit
            offload_folder="./offload_tmp"
        )
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
    def load_dataset(self):
        """Load LogiQA dataset using existing utils"""
        return load_logiqa_dataset()
            
    def evaluate_sample(self, example, max_samples=50):
        """Evaluate with memory management"""
        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            
        correct = 0
        results = []
        
        for i, ex in enumerate(dataset):
            # Clear memory every 10 examples
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
            context = ex.get('context', '')
            question = ex.get('question', '')
            options = ex.get('options', [])
            
            prompt = f"Context: {context}\nQuestion: {question}\nOptions:\n"
            for j, opt in enumerate(options):
                prompt += f"{chr(65+j)}. {opt}\n"
            prompt += "Answer:"
            
            # Tokenize with length limit
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024  # Reduced for memory
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Extract answer
            pred = response.upper().strip()
            if pred and pred[0] in 'ABCDE':
                predicted = pred[0]
            else:
                predicted = 'A'  # Default
                
            # Get correct answer
            correct_ans = ex.get('answer', 0)
            if isinstance(correct_ans, int):
                correct_ans = chr(65 + correct_ans)
                
            is_correct = predicted == correct_ans
            if is_correct:
                correct += 1
                
            results.append({
                'predicted': predicted,
                'correct': correct_ans,
                'is_correct': is_correct,
                'response': response
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(dataset)}, Accuracy: {correct/(i+1):.3f}")
                
        accuracy = correct / len(results)
        logger.info(f"Final Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
        
        return accuracy, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="results_4gb.json")
    args = parser.parse_args()
    
    evaluator = LogiQA4GB()
    evaluator.load_model()
    
    accuracy, results = evaluator.evaluate_sample(None, args.max_samples)
    
    # Save results
    output = {
        'accuracy': accuracy,
        'total': len(results),
        'correct': sum(1 for r in results if r['is_correct']),
        'results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Results saved to {args.output}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()