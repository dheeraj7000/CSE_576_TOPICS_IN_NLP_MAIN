#!/usr/bin/env python3
"""
config.py - FINAL MERGED VERSION

COMBINES:
1. Query config: CUDA optimization, cleaner structure, device handling
2. Uploaded config: Complete connector patterns, comprehensive settings
3. VALIDATED APPROACH: Multi-word connector support, architectural boost

TAG FORMAT: <connector type="X">word(s)</connector>
CONNECTOR TYPES: 6 categories with multi-word support
BOOST: 1.1× at input (multi-word support)
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """
    Configuration for connector-aware pretraining.
    OPTIMIZED FOR: Llama 3.2 3B on CUDA
    """
    
    # ========== MODEL CONFIGURATION ==========
    model_name: str = "meta-llama/Llama-3.2-3B"
    device: str = "cuda"  # "cuda", "cpu", "auto"
    torch_dtype: str = "bfloat16"  # Better stability for Llama
    
    # ========== CONNECTOR BOOSTING ==========
    use_connector_boost: bool = True
    boost_factor: float = 1.1  # Applied at input (multi-word support)
    boost_applies_to: str = "connector_words"  # Only connector words, not tags
    
    # ========== TAG FORMAT ==========
    # Supports both single and multi-word connectors
    tag_format: str = '<connector type="{type}">{word}</connector>'
    opening_tag_format: str = '<connector type="{type}">'
    closing_tag: str = '</connector>'
    
    # ========== LORA SETTINGS ==========
    use_lora: bool = False  # Llama 3.2 3B can do full fine-tuning
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # FFN
    ])
    
    # ========== CONNECTOR TYPES (COMPLETE) ==========
    # Each list contains single-word AND multi-word connector phrases
    connector_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'causal': [
            # Single-word
            'because', 'since', 'as', 'for', 'therefore', 'thus',
            'hence', 'consequently', 'accordingly', 'so',
            # Multi-word
            'as a result', 'as a consequence', 'for this reason',
            'that is why', 'thereby', 'wherefore', 'ergo',
            'given that', 'seeing that', 'in that', 'inasmuch as',
            'insofar as', 'on the grounds that', 'due to', 'owing to',
            'on account of', 'by virtue of', 'by reason of', 'thanks to',
            'so that', 'in order that', 'in order to', 'so as to',
            'with the aim of', 'for the purpose of', 'with a view to',
            'to the end that'
        ],
        'adversative': [
            # Single-word
            'but', 'however', 'yet', 'whereas', 'while', 'although',
            'though', 'despite', 'nonetheless',
            # Multi-word
            'on the contrary', 'in contrast', 'conversely', 'by contrast',
            'on the other hand', 'in opposition to', 'as opposed to',
            'rather than', 'unlike', 'even though', 'even if',
            'in spite of', 'notwithstanding', 'regardless of', 'admittedly',
            'granted that', 'granting that', 'be that as it may',
            'all the same', 'that said', 'instead', 'alternatively',
            'on the flip side', 'then again', 'otherwise', 'or else',
            'if not', 'nevertheless', 'still', 'anyway', 'anyhow',
            'at any rate', 'in any case', 'in any event', 'after all',
            'at the same time'
        ],
        'temporal': [
            # Single-word
            'when', 'while', 'as', 'whenever', 'before', 'after',
            'then', 'first', 'second', 'third', 'finally', 'last',
            'eventually', 'ultimately', 'during', 'throughout',
            # Multi-word
            'as long as', 'at the same time', 'simultaneously', 'meantime',
            'meanwhile', 'in the meantime', 'at that moment', 'at that time',
            'just then', 'prior to', 'previously', 'earlier', 'formerly',
            'beforehand', 'ahead of', 'in advance of', 'until', 'till',
            'up to', 'up until', 'afterwards', 'afterword', 'following',
            'subsequently', 'later', 'next', 'since', 'from then on',
            'thenceforth', 'thereafter', 'once', 'as soon as',
            'immediately after', 'firstly', 'secondly', 'thirdly',
            'in the end', 'to begin with', 'to start with', 'initially'
        ],
        'conditional': [
            # Single-word
            'if', 'when', 'whenever', 'once', 'unless',
            # Multi-word
            'assuming that', 'provided that', 'providing that',
            'given that', 'granted that', 'on condition that',
            'in the event that', 'in case', 'just in case', 'except if',
            'if not', 'only if', 'but for', 'save that',
            'were it not for', 'if it were not for'
        ],
        'conclusive': [
            # Single-word
            'therefore', 'thus', 'hence', 'so',
            # Multi-word
            'in conclusion', 'to conclude', 'in summary', 'to summarize',
            'to sum up', 'in sum', 'in short', 'in brief', 'all in all',
            'on the whole', 'overall', 'in overview', 'in general',
            'generally speaking', 'in other words', 'that is', 'that is to say',
            'namely', 'specifically', 'to put it another way', 'to rephrase',
            'i.e.', 'viz.', 'finally', 'last but not least',
            'in the final analysis', 'at last', 'ultimately', 'in the end'
        ],
        'additive': [
            # Single-word
            'and', 'also', 'too', 'plus',
            # Multi-word
            'as well', 'moreover', 'furthermore', 'in addition',
            'additionally', 'besides', 'what is more', 'on top of that',
            'along with', 'together with', 'coupled with', 'not to mention',
            'likewise', 'similarly', 'equally', 'in the same way',
            'by the same token', 'in like manner', 'correspondingly',
            'analogously', 'indeed', 'in fact', 'actually',
            'as a matter of fact', 'in truth', 'to tell the truth',
            'frankly', 'clearly', 'obviously', 'evidently', 'notably',
            'significantly', 'importantly', 'for example', 'for instance',
            'such as', 'namely', 'e.g.', 'to illustrate',
            'to demonstrate', 'as an illustration', 'in particular',
            'particularly', 'especially'
        ]
    })
    
    # ========== OPTIMIZATION SETTINGS ==========
    use_flash_attention: bool = False  # Not needed for 3B
    use_fsdp: bool = False  # Single GPU
    use_deepspeed: bool = False  # Single GPU
    gradient_checkpointing: bool = False  # 3B is small enough
    
    # ========== TRAINING CONFIGURATION ==========
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
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
    CHECKPOINT_DIR: str = "./checkpoints"
    
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
    dataloader_num_workers: int = 0  # CRITICAL: No multiprocessing
    dataloader_pin_memory: bool = False  # Safe for data loading
    dataloader_prefetch_factor: int = 2
    
    # ========== HELPER METHODS ==========
    
    def get_special_tokens(self) -> List[str]:
        """
        Get special tokens for tokenizer.
        
        Returns:
            List with opening tags (one per category) + closing tag
        """
        tokens = []
        
        # Opening tags
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
        assert self.boost_factor >= 1.0, "Boost factor must be >= 1.0"
        assert self.checkpoint_size > 0, "Checkpoint size must be positive"
        assert len(self.connector_types) > 0, "Must have at least one connector type"
        assert self.device in ["cuda", "cpu", "auto"], f"Invalid device: {self.device}"
        
        # Validate CUDA if needed
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("⚠️  WARNING: CUDA requested but not available!")
                print("    Falling back to CPU")
                self.device = "cpu"
        
        return True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
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
                print(f"CUDA: ✗ Not available")
        except:
            pass
        
        print(f"\nConnector Boosting:")
        print(f"  Enabled: {self.use_connector_boost}")
        print(f"  Factor: {self.boost_factor}×")
        print(f"  Applies to: {self.boost_applies_to}")
        print(f"  Multi-word support: YES")
        
        print(f"\nLoRA:")
        print(f"  Enabled: {self.use_lora}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_train_epochs}")
        print(f"  Batch size: {self.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.per_device_train_batch_size * self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        
        print(f"\nConnector Types: {len(self.connector_types)}")
        for conn_type, words in self.connector_types.items():
            print(f"  • {conn_type.upper()}: {len(words)} connector phrases (including multi-word)")
        
        print(f"\nSpecial Tokens: {len(self.get_special_tokens())}")
        print(f"Tag Format: {self.tag_format}")
        print(f"Checkpoint Directory: {self.checkpoint_dir}")
        print(f"Data Workers: {self.dataloader_num_workers} (NO MULTIPROCESSING)")
        print("=" * 70 + "\n")


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

_default_config = Config()

# Export constants
BASE_MODEL = _default_config.model_name
CONNECTOR_ATTENTION_WEIGHT = _default_config.boost_factor
MAX_SEQUENCE_LENGTH = _default_config.max_sequence_length
TAG_FORMAT = _default_config.tag_format
CHECKPOINT_DIR = _default_config.checkpoint_dir
SPECIAL_TOKENS = _default_config.get_special_tokens()


def get_connector_types() -> List[str]:
    """Get connector type names."""
    return _default_config.get_connector_type_names()


def verify_configuration():
    """Verify configuration."""
    print("\n" + "=" * 70)
    print("CONFIGURATION VERIFICATION - LLAMA-3.2-3B ON CUDA")
    print("=" * 70)
    
    config = _default_config
    config.validate()
    
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Device: {config.device}")
    print(f"✓ Tag format: {TAG_FORMAT}")
    print(f"✓ Boost factor: {config.boost_factor}x (multi-word support)")
    print(f"✓ LoRA: {'Enabled' if config.use_lora else 'Disabled'}")
    print(f"✓ Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"✓ Connector types: {len(config.connector_types)}")
    print(f"✓ Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"✓ Data workers: {config.dataloader_num_workers} (NO MULTIPROCESSING)")
    
    print("\nConnector types and multi-word support:")
    for ctype, words in config.connector_types.items():
        # Show some examples
        multi_word = [w for w in words if ' ' in w]
        print(f"  - {ctype.upper()}: {len(words)} phrases")
        if multi_word:
            print(f"    Examples: {', '.join(multi_word[:3])}...")
    
    print("\nSpecial tokens (first 3):")
    for token in SPECIAL_TOKENS[:3]:
        print(f"  • {token}")
    print(f"  ... and {len(SPECIAL_TOKENS) - 3} more")
    
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
