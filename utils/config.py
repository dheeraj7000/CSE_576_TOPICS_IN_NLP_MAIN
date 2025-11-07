#!/usr/bin/env python3
"""
config.py - MERGED & OPTIMIZED FOR LLAMA-3.2-3B ON CUDA

Combines best of both:
- Uploaded config.py: Complete connector patterns, discourse-aware settings
- Query config.py: CUDA optimization, cleaner structure

TAG FORMAT: <connector type="x">word</connector>
CONNECTOR TYPES: Causal, Adversative, Temporal, Conditional, Conclusive, Additive
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """
    Configuration class for connector-aware pretraining.
    OPTIMIZED FOR: Llama 3.2 3B on CUDA with discourse-aware features
    """
    
    # ========== MODEL CONFIGURATION (LLAMA 3.2 3B ON CUDA) ==========
    model_name: str = "meta-llama/Llama-3.2-3B"
    device: str = "cuda"  # Force CUDA
    torch_dtype: str = "bfloat16"  # Better stability for Llama
    
    # ========== CONNECTOR BOOSTING ==========
    use_connector_boost: bool = True
    boost_factor: float = 1.1  # Hidden state multiplication factor
    
    # ========== DISCOURSE-AWARE ATTENTION ==========
    use_discourse_attention: bool = True
    attention_modification_type: str = "pre_softmax"  # Apply scaling BEFORE softmax
    discourse_attention_scaling: float = 1.1  # Conservative weight
    connector_attention_weight: float = 1.1
    
    # ========== TAG FORMAT (CORRECT FORMAT) ==========
    tag_format: str = '<connector type="{type}">{word}</connector>'
    opening_tag_format: str = '<connector type="{type}">'
    closing_tag: str = '</connector>'
    
    # ========== LORA SETTINGS ==========
    use_lora: bool = False  # Llama 3.2 3B is small enough for full fine-tuning
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ])
    
    # ========== CONNECTOR TYPES (COMPLETE) ==========
    connector_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'causal': [
            'because', 'since', 'as', 'for', 'therefore', 'thus',
            'hence', 'consequently', 'accordingly', 'as a result',
            'due to', 'owing to', 'so that', 'in order to'
        ],
        'adversative': [
            'but', 'however', 'yet', 'whereas', 'while', 'although',
            'though', 'despite', 'in spite of', 'nevertheless',
            'nonetheless', 'on the other hand', 'in contrast'
        ],
        'temporal': [
            'when', 'while', 'as', 'whenever', 'before', 'after',
            'meanwhile', 'then', 'first', 'second', 'finally',
            'eventually', 'ultimately', 'during', 'throughout'
        ],
        'conditional': [
            'if', 'when', 'whenever', 'once', 'assuming',
            'provided that', 'given that', 'in case', 'unless'
        ],
        'conclusive': [
            'therefore', 'thus', 'hence', 'so', 'in conclusion',
            'to conclude', 'in summary', 'to summarize', 'in short',
            'overall', 'in general', 'finally', 'ultimately'
        ],
        'additive': [
            'and', 'also', 'too', 'moreover', 'furthermore',
            'in addition', 'besides', 'likewise', 'similarly',
            'for example', 'for instance', 'such as', 'particularly'
        ]
    })
    
    # ========== OPTIMIZATION SETTINGS ==========
    use_flash_attention: bool = True
    flash_attention_version: int = 2
    use_fsdp: bool = False  # Single GPU
    use_deepspeed: bool = False  # Single GPU
    gradient_checkpointing: bool = False  # Not needed for Llama 3.2 3B
    
    # ========== TRAINING CONFIGURATION (CUDA OPTIMIZED) ==========
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6  # Conservative for fine-tuning
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ========== SEQUENCE CONFIGURATION ==========
    max_sequence_length: int = 8192  # Llama 3.2 context
    max_length: int = 8192
    
    # ========== DATASET CONFIGURATION ==========
    combined_dataset_path: str = "./combined_dataset_sample"
    
    # ========== PREPROCESSING CONFIGURATION ==========
    checkpoint_size: int = 1000
    min_paper_length: int = 50
    max_paper_length: int = 500000
    checkpoint_dir: str = "./checkpoints"
    CHECKPOINT_DIR: str = "./checkpoints"  # For compatibility
    
    # ========== LOGGING ==========
    save_text_preview: bool = True
    preview_length: int = 500
    log_level: str = "INFO"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    
    # ========== PERFORMANCE ==========
    num_workers: int = 4
    batch_size: int = 10000
    dataloader_num_workers: int = 0  # CRITICAL: No multiprocessing for data loading
    dataloader_pin_memory: bool = False  # CRITICAL: False for safety
    dataloader_prefetch_factor: int = 2
    
    # ========== HELPER METHODS ==========
    
    def get_special_tokens(self) -> List[str]:
        """
        Get special tokens for tokenizer.
        
        Format: <connector type="TYPE">word</connector>
        
        Returns:
            List of special tokens
        """
        tokens = []
        
        # Opening tags (one per connector type)
        for conn_type in self.connector_types.keys():
            tokens.append(self.opening_tag_format.format(type=conn_type.upper()))
        
        # Closing tag
        tokens.append(self.closing_tag)
        
        return tokens
    
    def get_connector_type_names(self) -> List[str]:
        """Get connector type names (uppercase)."""
        return [conn_type.upper() for conn_type in self.connector_types.keys()]
    
    def validate(self) -> bool:
        """Validate configuration."""
        assert self.connector_attention_weight >= 1.0, "Attention weight must be >= 1.0"
        assert self.discourse_attention_scaling >= 1.0, "Attention scaling must be >= 1.0"
        assert self.checkpoint_size > 0, "Checkpoint size must be positive"
        assert len(self.connector_types) > 0, "Must have at least one connector type"
        assert self.boost_factor >= 1.0, "Boost factor must be >= 1.0"
        assert self.device in ["cuda", "cpu", "auto"], f"Invalid device: {self.device}"
        
        # Validate CUDA availability if device is "cuda"
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("⚠️  WARNING: CUDA device specified but not available!")
                print("    Falling back to CPU")
                self.device = "cpu"
        
        return True
    
    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return asdict(self)

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("CONFIGURATION SUMMARY - LLAMA-3.2-3B ON CUDA")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.torch_dtype}")
        
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA: ✓ Available")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print(f"CUDA: ✗ Not available (will use CPU)")
        except:
            pass
        
        print(f"\nConnector Boosting:")
        print(f"  Enabled: {self.use_connector_boost}")
        if self.use_connector_boost:
            print(f"  Boost factor: {self.boost_factor}x")
        
        print(f"\nDiscourse Attention:")
        print(f"  Enabled: {self.use_discourse_attention}")
        if self.use_discourse_attention:
            print(f"  Type: {self.attention_modification_type}")
            print(f"  Scaling: {self.discourse_attention_scaling}x")
        
        print(f"\nLoRA:")
        print(f"  Enabled: {self.use_lora}")
        if self.use_lora:
            print(f"  Rank: {self.lora_r}")
            print(f"  Alpha: {self.lora_alpha}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_train_epochs}")
        print(f"  Batch size: {self.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.per_device_train_batch_size * self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Max length: {self.max_length}")
        
        print(f"\nConnector Types: {len(self.connector_types)}")
        for conn_type, words in self.connector_types.items():
            print(f"  • {conn_type.upper()}: {len(words)} connectors")
        
        print(f"\nSpecial Tokens: {len(self.get_special_tokens())}")
        print(f"\nTag Format:")
        print(f"  Opening: {self.opening_tag_format}")
        print(f"  Closing: {self.closing_tag}")
        
        print(f"\nCheckpoint Directory: {self.checkpoint_dir}")
        print(f"Data Workers: {self.dataloader_num_workers}")
        print("=" * 70 + "\n")


# ============================================================================
# MODULE-LEVEL CONSTANTS (FOR BACKWARD COMPATIBILITY)
# ============================================================================

_default_config = Config()

# Export constants
BASE_MODEL = _default_config.model_name
CONNECTOR_ATTENTION_WEIGHT = _default_config.connector_attention_weight
DISCOURSE_ATTENTION_SCALING = _default_config.discourse_attention_scaling
MAX_SEQUENCE_LENGTH = _default_config.max_sequence_length
TAG_FORMAT = _default_config.tag_format
CHECKPOINT_DIR = _default_config.checkpoint_dir
SPECIAL_TOKENS = _default_config.get_special_tokens()


def get_connector_types() -> List[str]:
    """Get connector type names."""
    return _default_config.get_connector_type_names()


def verify_configuration():
    """Verify all configuration settings."""
    print("\n" + "=" * 70)
    print("CONFIGURATION VERIFICATION - LLAMA-3.2-3B ON CUDA")
    print("=" * 70)
    
    config = _default_config
    config.validate()
    
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Device: {config.device}")
    print(f"✓ Tag format: {TAG_FORMAT}")
    print(f"✓ Boost factor: {config.boost_factor}x")
    print(f"✓ Discourse attention: {config.use_discourse_attention}")
    print(f"✓ LoRA: {'Enabled' if config.use_lora else 'Disabled'}")
    print(f"✓ Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"✓ Connector types: {len(config.connector_types)}")
    print(f"✓ Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"✓ Data workers: {config.dataloader_num_workers} (NO MULTIPROCESSING)")
    
    print("\nConnector types:")
    for ctype, words in config.connector_types.items():
        print(f"  - {ctype.upper()}: {len(words)} words")
    
    print("\nSpecial tokens (first 5):")
    for token in SPECIAL_TOKENS[:5]:
        print(f"  • {token}")
    print(f"  ... and {len(SPECIAL_TOKENS) - 5} more")
    
    # Check CUDA
    try:
        import torch
        print("\nCUDA Status:")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ✗ CUDA not available (will use CPU)")
    except:
        print("\n⚠️  Could not check CUDA status")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    verify_configuration()