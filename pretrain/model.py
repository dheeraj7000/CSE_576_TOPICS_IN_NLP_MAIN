#!/usr/bin/env python3
"""
model_handler.py

Simplified model handler that works with single Config class.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class Model:

    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.original_vocab_size = None
    
    def load_tokenizer(self):
        print(f"Loading tokenizer: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.original_vocab_size = len(self.tokenizer)
        return self.tokenizer
    
    def extend_tokenizer(self, special_tokens):
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        return num_added
    
    def load_model(self):
        print(f"Loading model: {self.config.model_name}")

        dtype = getattr(torch, self.config.torch_dtype)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=self.config.device if self.config.device == "auto" else None,
            trust_remote_code=True
        )
        
        if self.config.device not in ["auto", None]:
            self.model = self.model.to(self.config.device)
        
        # Resize embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model
    
    def initialize_new_embeddings(self, connector_taxonomy):
        print("No embedding initialization logic set. ")
        
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "original_vocab_size": self.original_vocab_size,
            "current_vocab_size": len(self.tokenizer),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype)
        }
