#!/usr/bin/env python3
"""
Batch evaluation script for comparing different optimization methods
"""

import subprocess
import json
import time
from pathlib import Path
import argparse

def run_evaluation(mode, max_samples=None, model_name="meta-llama/Llama-3.2-3B"):
    """Run evaluation with specified mode"""
    print(f"\n{'='*50}")
    print(f"Running evaluation with {mode.upper()}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    cmd = ["python", "logiqa_baseline.py", "--model_name", model_name]
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    if mode == "fsdp":
        cmd.append("--use_fsdp")
    elif mode == "deepspeed":
        cmd = ["deepspeed", "logiqa_baseline.py", "--model_name", model_name, "--use_deepspeed"]
        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()
        
        print(f"✓ {mode.upper()} completed in {end_time - start_time:.2f} seconds")
        return True, end_time - start_time
        
    except subprocess.CalledProcessError as e:
        print(f"✗ {mode.upper()} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, 0

def compare_results():
    """Compare results from different methods"""
    results_dir = Path("./baseline_results")
    
    if not results_dir.exists():
        print("No results directory found")
        return
    
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    results = {}
    
    # Find result files
    for result_file in results_dir.glob("logiqa_results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            optimization = data.get('optimization', {})
            if optimization.get('use_fsdp'):
                method = "FSDP"
            elif optimization.get('use_deepspeed'):
                method = "DeepSpeed"
            else:
                method = "Simple"
                
            results[method] = {
                'accuracy': data.get('accuracy', 0),
                'total_examples': data.get('total_examples', 0),
                'correct_predictions': data.get('correct_predictions', 0)
            }
            
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Display comparison
    if results:
        print(f"{'Method':<12} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
        print("-" * 40)
        
        for method, data in sorted(results.items()):
            accuracy = data['accuracy']
            correct = data['correct_predictions']
            total = data['total_examples']
            print(f"{method:<12} {accuracy:<10.4f} {correct:<8} {total:<8}")
    else:
        print("No results found to compare")

def main():
    parser = argparse.ArgumentParser(description="Batch LogiQA Evaluation")
    parser.add_argument("--modes", nargs="+", default=["simple", "fsdp", "deepspeed"],
                       choices=["simple", "fsdp", "deepspeed"],
                       help="Evaluation modes to run")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples for testing")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B",
                       help="Model to evaluate")
    parser.add_argument("--compare_only", action="store_true",
                       help="Only compare existing results")
    
    args = parser.parse_args()
    
    if args.compare_only:
        compare_results()
        return
    
    print("Starting batch evaluation...")
    print(f"Modes: {args.modes}")
    print(f"Model: {args.model_name}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    
    results = {}
    
    for mode in args.modes:
        success, duration = run_evaluation(mode, args.max_samples, args.model_name)
        results[mode] = {'success': success, 'duration': duration}
    
    # Summary
    print(f"\n{'='*50}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    for mode, result in results.items():
        status = "✓" if result['success'] else "✗"
        duration = result['duration']
        print(f"{status} {mode.upper()}: {duration:.2f}s")
    
    # Compare results
    compare_results()

if __name__ == "__main__":
    main()