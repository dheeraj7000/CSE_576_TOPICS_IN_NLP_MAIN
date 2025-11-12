#!/usr/bin/env python3
"""
pretrain/model.py

Model wrapper for Llama 3.2 with connector-aware boosting.
Handles model loading, tokenizer extension, and embedding amplification.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

class ConnectorAwareModel:
    """Wrapper for Llama model with connector boosting capability"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*70}")
        print("MODEL INITIALIZATION")
        print(f"{'='*70}")
        
        # Load tokenizer (extended with connector tags)
        print(f"Loading tokenizer from: {cfg.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        print(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer):,})")
        
        # Load model
        print(f"\nLoading model: {cfg.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32,
            device_map="auto"
        )
        
        # Resize embeddings to match extended tokenizer
        original_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) != original_size:
            print(f"\nResizing model embeddings:")
            print(f"  Original: {original_size:,}")
            print(f"  New: {len(self.tokenizer):,}")
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.to(self.device)
        print(f"✓ Model loaded on {self.device}")
        print(f"{'='*70}\n")
    
    def get_num_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
