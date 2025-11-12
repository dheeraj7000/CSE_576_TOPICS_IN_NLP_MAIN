#!/usr/bin/env python3
"""
pretrain/utils.py

Utility functions for training, including memory management.
"""

import torch
import gc

def clear_cuda_memory():
    """
    Clear CUDA memory cache and invoke garbage collection.
    
    This helps prevent out-of-memory errors when running multiple training
    sessions or epochs without restarting the Python kernel.
    """
    if torch.cuda.is_available():
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Clear garbage in Python
        gc.collect()
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        print("✓ CUDA memory cleared")
    else:
        # Just clear Python garbage if CUDA not available
        gc.collect()
        print("✓ Python garbage collected (CUDA not available)")

def print_cuda_memory():
    """Print current CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n--- CUDA Memory Usage ---")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated:.2f} GB")
    else:
        print("CUDA not available")