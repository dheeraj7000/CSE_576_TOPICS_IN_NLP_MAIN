#!/usr/bin/env python3
"""
config.py - CORRECTED VERSION

Fixed: Proper connector tag formats that encode as SINGLE tokens.
These tags ARE added to tokenizer vocab as special tokens.
"""

import os
from typing import List, Dict
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """Configuration for connector-aware pretraining."""
    
    # Model Configuration
    model_name: str = "meta-llama/Llama-3.2-3B"
    tokenizer_path: str = "./tokenizer_extended"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    
    data_dir: str = "data_splits"
    files_per_chunk: int = 5
    batch_size: int = 128
    train_pattern: str = "train_chunk_*.parquet"
    max_batches_to_print: int = 20

    # Connector Boosting
    use_connector_boost: bool = True
    boost_factor: float = 1.1
    boost_applies_to: str = "connector_words"
    
    # Tag Format - CORRECTED: Proper format tags
    tag_format: str = '<connector type="{type}">{word}</connector>'
    opening_tag_format: str = '<connector type="{type}">'
    closing_tag: str = '</connector>'
    
    # LoRA Settings
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Connector Types with Multi-word Support
    connector_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'CAUSAL': [
            'because', 'since', 'as', 'for', 'therefore', 'thus',
            'hence', 'consequently', 'accordingly', 'so',
            'as a result', 'as a consequence', 'for this reason',
            'that is why', 'thereby', 'wherefore', 'ergo',
            'given that', 'seeing that', 'in that', 'inasmuch as',
            'insofar as', 'on the grounds that', 'due to', 'owing to',
            'on account of', 'by virtue of', 'by reason of', 'thanks to',
            'so that', 'in order that', 'in order to', 'so as to',
            'with the aim of', 'for the purpose of', 'with a view to',
            'to the end that'
        ],
        'ADVERSATIVE': [
            'but', 'however', 'yet', 'whereas', 'while', 'although',
            'though', 'despite', 'nonetheless',
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
        'TEMPORAL': [
            'when', 'while', 'as', 'whenever', 'before', 'after',
            'then', 'first', 'second', 'third', 'finally', 'last',
            'eventually', 'ultimately', 'during', 'throughout',
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
        'CONDITIONAL': [
            'if', 'when', 'whenever', 'once', 'unless',
            'assuming that', 'provided that', 'providing that',
            'given that', 'granted that', 'on condition that',
            'in the event that', 'in case', 'just in case', 'except if',
            'if not', 'only if', 'but for', 'save that',
            'were it not for', 'if it were not for'
        ],
        'CONCLUSIVE': [
            'therefore', 'thus', 'hence', 'so',
            'in conclusion', 'to conclude', 'in summary', 'to summarize',
            'to sum up', 'in sum', 'in short', 'in brief', 'all in all',
            'on the whole', 'overall', 'in overview', 'in general',
            'generally speaking', 'in other words', 'that is', 'that is to say',
            'namely', 'specifically', 'to put it another way', 'to rephrase',
            'i.e.', 'viz.', 'finally', 'last but not least',
            'in the final analysis', 'at last', 'ultimately', 'in the end'
        ],
        'ADDITIVE': [
            'and', 'also', 'too', 'plus',
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
    
    # Optimization Settings
    use_flash_attention: bool = False
    use_fsdp: bool = False
    use_deepspeed: bool = False
    gradient_checkpointing: bool = False
    
    # Training Configuration
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Sequence Configuration
    max_sequence_length: int = 8192
    max_length: int = 8192
    
    # Dataset Configuration
    combined_dataset_path: str = "./combined_dataset_sample"
    
    # Preprocessing Configuration
    checkpoint_size: int = 1000
    min_paper_length: int = 50
    max_paper_length: int = 500000
    checkpoint_dir: str = "./checkpoints"
    CHECKPOINT_DIR: str = "./checkpoints"  # Backwards compatibility

    # Checkpoint frequency (save every N files processed)
    checkpoint_frequency: int = 20

    # Checkpoint metadata (optional)
    save_optimizer_state: bool = False  # Set to True to save optimizer state
    save_full_state: bool = False       # Set to True to save training state
    
    # Logging
    save_text_preview: bool = True
    preview_length: int = 500
    log_level: str = "INFO"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Performance - OPTIMIZED
    num_workers: int = 4
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    
    # Helper Methods
    def get_special_tokens(self) -> List[str]:
        """Get special tokens for tokenizer."""
        tokens = []
        for conn_type in self.connector_types.keys():
            tokens.append(self.opening_tag_format.format(type=conn_type.upper()))
        tokens.append(self.closing_tag)
        return tokens
    
    def get_connector_type_names(self) -> List[str]:
        """Get connector type names."""
        return [conn_type.upper() for conn_type in self.connector_types.keys()]
    
    def validate(self) -> bool:
        """Validate configuration."""
        assert self.boost_factor >= 1.0, "Boost factor must be >= 1.0"
        assert self.checkpoint_size > 0, "Checkpoint size must be positive"
        assert len(self.connector_types) > 0, "Must have at least one connector type"
        assert self.device in ["cuda", "cpu", "auto"], f"Invalid device: {self.device}"
        
        # Validate tag format
        assert self.opening_tag_format, "opening_tag_format cannot be empty!"
        assert self.closing_tag, "closing_tag cannot be empty!"
        
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                print("⚠️  WARNING: CUDA requested but not available!")
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
        
        print(f"\nTag Format:")
        print(f"  Opening: {self.opening_tag_format}")
        print(f"  Closing: {self.closing_tag}")
        print(f"  Full:    {self.tag_format}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.num_train_epochs}")
        print(f"  Batch size: {self.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.per_device_train_batch_size * self.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.learning_rate}")
        
        print(f"\nConnector Types: {len(self.connector_types)}")
        for conn_type, words in self.connector_types.items():
            print(f"  • {conn_type.upper()}: {len(words)} phrases")
        
        print(f"\nSpecial Tokens: {len(self.get_special_tokens())}")
        tokens = self.get_special_tokens()
        print(f"  Sample: {tokens[:2]}")
        
        print(f"\nData Loading:")
        print(f"  Workers: {self.dataloader_num_workers} (parallel)")
        print(f"  Pin memory: {self.dataloader_pin_memory}")

        print(f"\nCheckpoint System:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Frequency: Every {self.checkpoint_frequency} files")
        print(f"  Save optimizer: {self.save_optimizer_state}")
        print(f"  Save full state: {self.save_full_state}")
                
        print("=" * 70 + "\n")


# Module-level exports
_default_config = Config()
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
    print("CONFIGURATION VERIFICATION - LLAMA-3.2-3B")
    print("=" * 70)
    
    config = _default_config
    config.validate()
    
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Device: {config.device}")
    print(f"✓ Opening tag format: {config.opening_tag_format}")
    print(f"✓ Closing tag format: {config.closing_tag}")
    print(f"✓ Boost factor: {config.boost_factor}x")
    print(f"✓ Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"✓ Connector types: {len(config.connector_types)}")
    
    print(f"\nSpecial tokens:")
    for i, token in enumerate(SPECIAL_TOKENS, 1):
        print(f"  {i}. {token}")
    
    print(f"\nConnector types:")
    for ctype, words in config.connector_types.items():
        print(f"  - {ctype.upper()}: {len(words)} phrases")
    
    try:
        import torch
        print(f"\nCUDA Status:")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ✗ CUDA not available")
    except:
        pass
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    verify_configuration()
