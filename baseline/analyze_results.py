#!/usr/bin/env python3
"""
Analyze LogiQA baseline results
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def load_results(results_file):
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_accuracy(results):
    """Analyze overall accuracy"""
    total = results['total_examples']
    correct = results['correct_predictions']
    accuracy = results['accuracy']
    
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Error Rate: {1-accuracy:.4f}")

def analyze_answer_distribution(results):
    """Analyze answer choice distribution"""
    correct_answers = []
    predicted_answers = []
    
    for result in results['results']:
        correct_answers.append(result['correct_answer'])
        predicted_answers.append(result['predicted_answer'])
    
    print(f"\nCorrect Answer Distribution:")
    correct_dist = Counter(correct_answers)
    for choice in sorted(correct_dist.keys()):
        count = correct_dist[choice]
        pct = 100 * count / len(correct_answers)
        print(f"  {choice}: {count} ({pct:.1f}%)")
    
    print(f"\nPredicted Answer Distribution:")
    pred_dist = Counter(predicted_answers)
    for choice in sorted(pred_dist.keys()):
        count = pred_dist[choice]
        pct = 100 * count / len(predicted_answers)
        print(f"  {choice}: {count} ({pct:.1f}%)")

def analyze_by_correct_answer(results):
    """Analyze accuracy by correct answer choice"""
    by_answer = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for result in results['results']:
        correct_ans = result['correct_answer']
        is_correct = result['is_correct']
        
        by_answer[correct_ans]['total'] += 1
        if is_correct:
            by_answer[correct_ans]['correct'] += 1
    
    print(f"\nAccuracy by Correct Answer:")
    for choice in sorted(by_answer.keys()):
        data = by_answer[choice]
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"  {choice}: {accuracy:.4f} ({data['correct']}/{data['total']})")

def analyze_response_patterns(results):
    """Analyze response patterns"""
    response_lengths = []
    unknown_responses = 0
    
    for result in results['results']:
        response = result['response']
        response_lengths.append(len(response))
        
        if result['predicted_answer'] == 'UNKNOWN':
            unknown_responses += 1
    
    print(f"\nResponse Analysis:")
    print(f"Average response length: {sum(response_lengths)/len(response_lengths):.1f} chars")
    print(f"Unknown responses: {unknown_responses} ({100*unknown_responses/len(results['results']):.1f}%)")

def show_error_examples(results, num_examples=5):
    """Show examples of incorrect predictions"""
    errors = [r for r in results['results'] if not r['is_correct']]
    
    print(f"\nError Examples (showing {min(num_examples, len(errors))}):")
    print("-" * 80)
    
    for i, error in enumerate(errors[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"Context: {error['context'][:100]}...")
        print(f"Question: {error['question']}")
        print(f"Options: {error['options']}")
        print(f"Correct: {error['correct_answer']}")
        print(f"Predicted: {error['predicted_answer']}")
        print(f"Response: '{error['response']}'")

def main():
    parser = argparse.ArgumentParser(description="Analyze LogiQA Results")
    parser.add_argument("results_file", type=str, 
                       help="Path to results JSON file")
    parser.add_argument("--show_errors", type=int, default=3,
                       help="Number of error examples to show")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Results file not found: {args.results_file}")
        return
    
    # Load results
    results = load_results(args.results_file)
    
    print("LogiQA Results Analysis")
    print("=" * 50)
    print(f"Model: {results['model_name']}")
    
    # Optimization info
    opt = results['optimization']
    opt_mode = "FSDP" if opt['use_fsdp'] else "DeepSpeed" if opt['use_deepspeed'] else "Simple"
    print(f"Optimization: {opt_mode}")
    
    print("=" * 50)
    
    # Run analyses
    analyze_accuracy(results)
    analyze_answer_distribution(results)
    analyze_by_correct_answer(results)
    analyze_response_patterns(results)
    
    if args.show_errors > 0:
        show_error_examples(results, args.show_errors)

if __name__ == "__main__":
    main()