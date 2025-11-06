#!/usr/bin/env python3
"""
model_handler.py

Simplified model handler with connector boosting support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class Model:
    """Model handler with optional connector boosting."""

    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.original_vocab_size = None
        
        # NEW: Connector boosting settings
        self.use_connector_boost = getattr(config, 'use_connector_boost', False)
        self.boost_factor = getattr(config, 'boost_factor', 1.1)
    
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
        
        # NEW: Wrap model with connector boosting if enabled
        if self.use_connector_boost:
            self.model = self._wrap_with_connector_boost(self.model)
            print(f"✓ Connector boosting enabled (factor={self.boost_factor})")
        
        return self.model
    
    def _wrap_with_connector_boost(self, model):
        """
        NEW: Wrap the model to apply connector boosting.
        
        This modifies the forward pass to apply boosting to hidden states
        at connector token positions.
        """
        original_forward = model.forward
        boost_factor = self.boost_factor
        
        def forward_with_boost(
            input_ids=None,
            attention_mask=None,
            connector_mask=None,  # NEW parameter
            **kwargs
        ):
            """
            Modified forward pass with connector boosting.
            
            Args:
                connector_mask: [batch, seq_len] with values 1.0 or boost_factor
            """
            # If no connector mask provided, create default (all 1.0)
            if connector_mask is None:
                connector_mask = torch.ones_like(input_ids, dtype=torch.float)
            
            # Store connector mask for use in transformer blocks
            model._connector_mask = connector_mask
            
            # Call original forward
            outputs = original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Clean up
            if hasattr(model, '_connector_mask'):
                delattr(model, '_connector_mask')
            
            return outputs
        
        # Replace forward method
        model.forward = forward_with_boost
        
        # Inject boosting into transformer blocks
        self._inject_boosting_into_blocks(model, boost_factor)
        
        return model
    
    def _inject_boosting_into_blocks(self, model, boost_factor):
        """
        NEW: Inject connector boosting into transformer blocks.
        
        This wraps each transformer block to apply boosting after
        attention and feed-forward layers.
        """
        # Try to find transformer blocks (works for most architectures)
        block_containers = [
            'model.layers',        # Llama, Mistral
            'transformer.h',       # GPT-2
            'gpt_neox.layers',     # GPT-NeoX
            'model.decoder.layers' # OPT
        ]
        
        blocks = None
        for container_path in block_containers:
            try:
                blocks = self._get_nested_attr(model, container_path)
                if blocks is not None:
                    print(f"   Found transformer blocks at: {container_path}")
                    break
            except:
                continue
        
        if blocks is None:
            print("   ⚠ Warning: Could not find transformer blocks, boosting may not work")
            return
        
        # Wrap each block
        for i, block in enumerate(blocks):
            original_block_forward = block.forward
            
            def create_boosted_forward(original_forward):
                def boosted_forward(hidden_states, *args, **kwargs):
                    # Get connector mask from model
                    connector_mask = getattr(model, '_connector_mask', None)
                    
                    # Call original block forward
                    outputs = original_forward(hidden_states, *args, **kwargs)
                    
                    # Apply boosting to hidden states
                    if connector_mask is not None:
                        if isinstance(outputs, tuple):
                            # Extract hidden states (usually first element)
                            boosted_hidden = outputs[0]
                            
                            # Apply boost: [batch, seq, hidden] * [batch, seq, 1]
                            boost = connector_mask.unsqueeze(-1).to(boosted_hidden.device)
                            boosted_hidden = boosted_hidden * boost
                            
                            # Reconstruct outputs
                            outputs = (boosted_hidden,) + outputs[1:]
                        else:
                            # If output is not tuple, boost directly
                            boost = connector_mask.unsqueeze(-1).to(outputs.device)
                            outputs = outputs * boost
                    
                    return outputs
                
                return boosted_forward
            
            # Replace block's forward method
            block.forward = create_boosted_forward(original_block_forward)
    
    def _get_nested_attr(self, obj, attr_path):
        """Helper to get nested attributes like 'model.layers'"""
        attrs = attr_path.split('.')
        for attr in attrs:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                return None
        return obj
    
    def initialize_new_embeddings(self, connector_taxonomy):
        print("No embedding initialization logic set.")
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "original_vocab_size": self.original_vocab_size,
            "current_vocab_size": len(self.tokenizer),
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            # NEW: Boosting info
            "connector_boosting": self.use_connector_boost,
            "boost_factor": self.boost_factor if self.use_connector_boost else None
        }
        
        return info
