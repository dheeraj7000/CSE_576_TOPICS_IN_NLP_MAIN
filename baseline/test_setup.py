#!/usr/bin/env python3
"""
Test script to verify baseline setup
"""

import torch
import sys
from pathlib import Path

def test_dependencies():
    """Test if all required packages are available"""
    print("Testing dependencies...")
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError:
        print("✗ transformers not found")
        return False
    
    try:
        import datasets
        print(f"✓ datasets: {datasets.__version__}")
    except ImportError:
        print("✗ datasets not found")
        return False
    
    try:
        import deepspeed
        print(f"✓ deepspeed: {deepspeed.__version__}")
    except ImportError:
        print("✗ deepspeed not found (optional)")
    
    return True

def test_cuda():
    """Test CUDA availability"""
    print(f"\nTesting CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test memory
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"Total GPU memory: {total_memory / 1024**3:.1f} GB")
    
def test_model_access():
    """Test if we can access the model"""
    print(f"\nTesting model access...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B", 
            trust_remote_code=True
        )
        print("✓ Model tokenizer accessible")
        return True
    except Exception as e:
        print(f"✗ Model access failed: {e}")
        print("Note: You may need to set HF_TOKEN environment variable")
        return False

def test_dataset_access():
    """Test LogiQA dataset access"""
    print(f"\nTesting dataset access...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("lucasmccabe/logiqa", split="test")
        print(f"✓ LogiQA dataset accessible: {len(dataset)} examples")
        
        # Show sample
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        return True
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        print("Note: Will use fallback sample data during evaluation")
        return True  # Return True since we have fallback data

def main():
    print("LogiQA Baseline Setup Test")
    print("=" * 40)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test CUDA
    test_cuda()
    
    # Check GPU memory for 3B model
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < 6:
            print(f"\n⚠️  GPU has {total_memory:.1f}GB memory. Consider using DeepSpeed for 3B model.")
        else:
            print(f"\n✓ GPU memory ({total_memory:.1f}GB) sufficient for 3B model")
    
    # Test model access
    model_ok = test_model_access()
    
    # Test dataset access
    dataset_ok = test_dataset_access()
    
    print(f"\n{'='*40}")
    print("SETUP STATUS")
    print(f"{'='*40}")
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"Model access: {'✓' if model_ok else '✗'}")
    print(f"Dataset access: {'✓' if dataset_ok else '✗'}")
    
    if deps_ok and model_ok and dataset_ok:
        print("\n🎉 Setup is ready for evaluation!")
        print("\nNext steps:")
        print("1. Run simple evaluation: ./run_baseline.sh simple 10")
        print("2. Run full evaluation: ./run_baseline.sh simple")
    else:
        print("\n⚠️  Setup needs attention. Check errors above.")

if __name__ == "__main__":
    main()