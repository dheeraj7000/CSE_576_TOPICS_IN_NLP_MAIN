#!/usr/bin/env python3

from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class Config:   
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"  # Options: "gpt2", "meta-llama/Llama-3.1-8B", "microsoft/phi-4-mini"
    device: str = "cuda"  # "auto", "cuda", "cpu"
    torch_dtype: str = "float16"  # "auto", "float32", "float16", "bfloat16"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    
    # ========================================================================
    # CONNECTOR SETTINGS
    # ========================================================================
    
    # Special tokens that will be added to tokenizer
    connector_start: str = "<connector>"
    connector_end: str = "</connector>"
    
    # Connector types and example words
    # Add more types or words as needed
    connector_types: dict = field(default_factory=lambda: {
        'conclusive': ['therefore', 'thus', 'hence', 'consequently'],
        'contrastive': ['however', 'although', 'but', 'yet'],
        'causal': ['because', 'since', 'due to'],
        'temporal': ['meanwhile', 'then', 'after', 'before'],
        'additive': ['furthermore', 'moreover', 'also']
    })
    
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_special_tokens(self) -> List[str]:
        """Get list of special tokens to add to tokenizer"""
        tokens = [self.connector_start, self.connector_end]
        for conn_type in self.connector_types.keys():
            tokens.append(f'type="{conn_type}"')
        return tokens
