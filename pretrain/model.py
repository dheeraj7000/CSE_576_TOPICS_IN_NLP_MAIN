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
        embedding_layer = self.model.get_input_embeddings()
        current_vocab_size = len(self.tokenizer)
        
        with torch.no_grad():
            existing_embeddings = embedding_layer.weight[:self.original_vocab_size].clone()
            
            for token_id in range(self.original_vocab_size, current_vocab_size):
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                initialized = False
                
                # Try to match connector type
                for conn_type, example_words in connector_taxonomy.items():
                    if conn_type.lower() in str(token).lower():
                        similar_ids = []
                        for word in example_words[:5]:
                            word_tokens = self.tokenizer.tokenize(word)
                            if word_tokens:
                                word_id = self.tokenizer.convert_tokens_to_ids(word_tokens)
                                if word_id < self.original_vocab_size:
                                    similar_ids.append(word_id)
                        
                        if similar_ids:
                            avg_embedding = existing_embeddings[similar_ids].mean(dim=0)
                            embedding_layer.weight.data[token_id] = avg_embedding
                            initialized = True
                            break
                
                if not initialized:
                    embedding_layer.weight.data[token_id] = existing_embeddings.mean(dim=0)
                    print(f"'{token}' initialized with mean embedding")
    
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
