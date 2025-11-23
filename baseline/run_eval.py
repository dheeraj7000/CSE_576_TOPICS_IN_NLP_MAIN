import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from tqdm import tqdm
import argparse
import requests

class LogiQAEvaluator:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", batch_size=16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size}")
        
        # Load tokenizer and model
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        
        self.tokenizer.padding_side = "left"
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset using the new method
        self.dataset = self.load_logiqa_dataset()
        
    def load_logiqa_dataset(self):
        """Load LogiQA dataset from raw GitHub files"""
        urls = {
            "train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
            "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt", 
            "test": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt"
        }
        
        dataset_dict = {}
        
        for split, url in urls.items():
            print(f"Downloading {split} split...")
            response = requests.get(url)
            lines = response.text.split('\n')
            
            examples = []
            for i in range(0, len(lines), 8):
                if i + 7 >= len(lines):
                    break
                    
                # Parse the dataset format
                correct_answer = lines[i].strip().replace('.', '')
                context = lines[i+1].strip()
                question = lines[i+2].strip()
                options = [lines[i+3].strip(), lines[i+4].strip(), 
                          lines[i+5].strip(), lines[i+6].strip()]
                
                # Convert answer from 'a', 'b', 'c', 'd' to 0, 1, 2, 3
                answer_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
                correct_option = answer_map.get(correct_answer.lower(), 0)
                
                # Convert to letter format for consistency
                answer_letter = ['A', 'B', 'C', 'D'][correct_option]
                
                # Clean options (remove 'A. ', 'B. ', etc.)
                cleaned_options = []
                for opt in options:
                    if len(opt) > 2 and opt[1] == '.' and opt[0] in 'ABCD':
                        cleaned_options.append(opt[3:].strip())
                    else:
                        cleaned_options.append(opt.strip())
                
                examples.append({
                    'context': context,
                    'question': question,
                    'options': cleaned_options,
                    'answer': answer_letter,
                    'correct_option': correct_option
                })
            
            dataset_dict[split] = examples
            print(f"Loaded {len(examples)} examples for {split} split")
        
        return dataset_dict
    
    def create_prompt(self, example):
        """Create a prompt for the LogiQA question"""
        prompt = f"""You are a logical reasoning assistant. Analyze the following question and options, then provide your answer.

Context: {example['context']}

Question: {example['question']}

Options:
A) {example['options'][0]}
B) {example['options'][1]}
C) {example['options'][2]}
D) {example['options'][3]}

Please think step by step and provide your final answer in the format: "Answer: X" where X is A, B, C, or D.

Reasoning:"""
        return prompt
    
    def create_batch_prompts(self, examples):
        """Create prompts for a batch of examples"""
        return [self.create_prompt(ex) for ex in examples]
    
    def extract_answer(self, text):
        """Extract the answer from model output"""
        # Look for patterns like "Answer: A", "Answer is B", etc.
        patterns = [
            r"Answer:\s*([ABCD])",
            r"Answer\s*is\s*([ABCD])",
            r"Final Answer:\s*([ABCD])",
            r"Option\s*([ABCD])",
            r"\b([ABCD])\b.*$"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no clear pattern found, return the last occurrence of A, B, C, or D
        matches = re.findall(r'[ABCD]', text.upper())
        if matches:
            return matches[-1]
        
        return None
    
    def generate_batch_answers(self, examples, max_new_tokens=256):
        """Generate answers for a batch of examples"""
        prompts = self.create_batch_prompts(examples)
        
        # Tokenize all prompts at once
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate responses for the entire batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,  # This includes input_ids and attention_mask
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        results = []
        for i, (output, example, prompt) in enumerate(zip(outputs, examples, prompts)):
            # Decode response
            response = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            # Extract answer
            predicted_answer = self.extract_answer(generated_text)
            
            results.append({
                'id': hash(example['context'] + example['question']),
                'prompt': prompt,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'correct_answer': example['answer'],
                'is_correct': predicted_answer == example['answer'],
                'context': example['context'],
                'question': example['question'],
                'options': example['options']
            })
        
        return results
    
    def evaluate_split(self, split="test", num_samples=None):
        """Evaluate the model on a specific split using batch processing"""
        if split not in self.dataset:
            raise ValueError(f"Split {split} not found. Available splits: {list(self.dataset.keys())}")
        
        examples = self.dataset[split]
        
        if num_samples:
            examples = examples[:num_samples]
        
        results = []
        correct_count = 0
        
        print(f"Evaluating on {len(examples)} samples from {split} split...")
        print(f"Using batch size: {self.batch_size}")
        
        # Process in batches
        for batch_start in tqdm(range(0, len(examples), self.batch_size)):
            batch_end = min(batch_start + self.batch_size, len(examples))
            batch_examples = examples[batch_start:batch_end]
            
            try:
                batch_results = self.generate_batch_answers(batch_examples)
                results.extend(batch_results)
                
                # Update correct count
                batch_correct = sum(1 for r in batch_results if r['is_correct'])
                correct_count += batch_correct
                
                # Print progress
                if (batch_end // self.batch_size) % 5 == 0:  # Print every 5 batches
                    accuracy = correct_count / batch_end
                    print(f"Progress: {batch_end}/{len(examples)}, Current Accuracy: {accuracy:.4f}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # Process individually if batch fails
                for i, example in enumerate(batch_examples):
                    try:
                        result = self.generate_batch_answers([example])[0]  # Single item batch
                        results.append(result)
                        if result['is_correct']:
                            correct_count += 1
                    except Exception as e2:
                        print(f"Error processing individual example {batch_start + i}: {e2}")
                        results.append({
                            'id': batch_start + i,
                            'prompt': self.create_prompt(example),
                            'generated_text': "",
                            'predicted_answer': None,
                            'correct_answer': example['answer'],
                            'is_correct': False,
                            'error': str(e2),
                            'context': example['context'],
                            'question': example['question'],
                            'options': example['options']
                        })
        
        # Calculate final metrics
        accuracy = correct_count / len(results)
        
        metrics = {
            "split": split,
            "total_samples": len(results),
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "model_name": "llama-3.2-3b-instruct",
            "batch_size": self.batch_size
        }
        
        return results, metrics
    
    def save_results(self, results, metrics, output_file="logiqa_results.json"):
        """Save results to JSON file"""
        output_data = {
            "metrics": metrics,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_file}")
        
    def print_detailed_analysis(self, results, metrics):
        """Print detailed analysis of results"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Model: {metrics['model_name']}")
        print(f"Split: {metrics['split']}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Correct Predictions: {metrics['correct_predictions']}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Batch Size: {metrics['batch_size']}")
        
        # Analyze answer distribution
        answer_counts = {}
        correct_by_answer = {}
        
        for result in results:
            correct_answer = result['correct_answer']
            predicted_answer = result['predicted_answer']
            
            # Count correct answers
            if correct_answer not in correct_by_answer:
                correct_by_answer[correct_answer] = {'total': 0, 'correct': 0}
            correct_by_answer[correct_answer]['total'] += 1
            if result['is_correct']:
                correct_by_answer[correct_answer]['correct'] += 1
            
            # Count predictions
            if predicted_answer not in answer_counts:
                answer_counts[predicted_answer] = 0
            answer_counts[predicted_answer] += 1
        
        print(f"\nAnswer Distribution:")
        for answer in ['A', 'B', 'C', 'D']:
            count = answer_counts.get(answer, 0)
            print(f"  {answer}: {count}")
        
        print(f"\nAccuracy by Correct Answer:")
        for answer in ['A', 'B', 'C', 'D']:
            if answer in correct_by_answer:
                stats = correct_by_answer[answer]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  {answer}: {stats['correct']}/{stats['total']} ({acc:.4f})")
        
        # Show some examples
        print(f"\nExample Predictions:")
        correct_examples = [r for r in results if r['is_correct']]
        incorrect_examples = [r for r in results if not r['is_correct'] and r['predicted_answer'] is not None]
        
        if correct_examples:
            print(f"\nCorrect Example:")
            self.print_example(correct_examples[0])
        
        if incorrect_examples:
            print(f"\nIncorrect Example:")
            self.print_example(incorrect_examples[0])
    
    def print_example(self, result):
        """Print a single example result"""
        print(f"Context: {result['context'][:100]}...")
        print(f"Question: {result['question']}")
        print(f"Generated: {result['generated_text'][:200]}...")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Correct: {result['correct_answer']}")
        print(f"Match: {result['is_correct']}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama3.2 3B on LogiQA dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--output", type=str, default="logiqa_results.json", help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Initialize evaluator with larger batch size
    print("Loading model and dataset...")
    evaluator = LogiQAEvaluator(batch_size=args.batch_size)
    
    # Run evaluation
    results, metrics = evaluator.evaluate_split(
        split=args.split,
        num_samples=args.num_samples
    )
    
    # Save results
    evaluator.save_results(results, metrics, args.output)
    
    # Print analysis
    evaluator.print_detailed_analysis(results, metrics)

if __name__ == "__main__":
    main()