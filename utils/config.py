#!/usr/bin/env python3
"""
config.py - MERGED: DISCOURSE-AWARE REASONING WITH YOUR TAG FORMAT

Combines:
- Your exact tag format: <connector type="X"> word </connector>
- Your connector types (causal, adversative, temporal, conditional, conclusive, additive)
- Architectural attention modification (pre-softmax, conservative 1.1x weight)
- Loss weighting integration
- FSDP, DeepSpeed, Flash Attention optimization

Compatible with: preprocess.py, discourse_training.py, discourse_aware_model.py
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# ============================================================================
# CONFIG CLASS - MERGED WITH DISCOURSE AWARENESS
# ============================================================================

@dataclass
class Config:
    """
    Configuration class for discourse-aware reasoning pretraining.
    
    Integrates your existing settings with architectural attention modifications.
    """
    
    # ========== MODEL CONFIGURATION ==========
    model_name: str = "meta-llama/Llama-3.2-3B"
    device: str = "auto"  # "auto", "cuda", "cpu"
    torch_dtype: str = "bfloat16"  # bfloat16 recommended for stability
    
    # ========== DISCOURSE-AWARE ATTENTION (NEW) ==========
    # Architectural modification
    use_discourse_attention: bool = True
    attention_modification_type: str = "pre_softmax"  # Apply scaling BEFORE softmax
    discourse_attention_scaling: float = 1.1  # Conservative weight (pre-softmax)
    
    # Loss weighting
    connector_attention_weight: float = 1.1  # Conservative: won't over-prioritize
    
    # ========== TAG FORMAT (YOUR EXACT FORMAT) ==========
    tag_format: str = '<connector type="{type}"> {word} </connector>'
    
    # ========== LORA SETTINGS ==========
    use_lora: bool = True  # Enable LoRA for efficiency
    lora_r: int = 32  # Increased for reasoning improvement
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    
    # LoRA target modules (your FFN targets + attention)
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj"      # FFN for reasoning
    ])
    
    # ========== CONNECTOR TYPES (YOUR EXACT TYPES) ==========
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
    # Flash Attention
    use_flash_attention: bool = True
    flash_attention_version: int = 2
    
    # FSDP/DeepSpeed
    use_fsdp: bool = False
    use_deepspeed: bool = False
    deepspeed_config: str = "./deepspeed_config_discourse.json"
    
    # ========== TRAINING CONFIGURATION ==========
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.15
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ========== SEQUENCE CONFIGURATION ==========
    max_sequence_length: int = 128000  # Llama 3.2 max context
    
    # ========== DATASET CONFIGURATION ==========
    combined_dataset_path: str = "./combined_dataset_sample"
    
    # ========== PREPROCESSING CONFIGURATION ==========
    checkpoint_size: int = 1000
    min_paper_length: int = 50
    max_paper_length: int = 500000
    checkpoint_dir: str = "./checkpoints"
    
    # ========== LOGGING ==========
    save_text_preview: bool = True
    preview_length: int = 1000
    log_level: str = "INFO"
    
    # ========== PERFORMANCE ==========
    num_workers: int = 4
    batch_size: int = 10000
    
    # ========== HELPER METHODS ==========
    
    def get_special_tokens(self) -> List[str]:
        """
        Get special tokens for your exact tag format.
        
        Format: <connector type="X"> word </connector>
        
        Returns tokens for tagging.
        """
        tokens = []
        
        # Opening tags (one per connector type)
        for conn_type in self.connector_types.keys():
            tokens.append(f'<connector type="{conn_type}">')
        
        # Closing tag
        tokens.append('</connector>')
        
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
        return True


# ============================================================================
# MODULE-LEVEL CONSTANTS (BACKWARDS COMPATIBLE)
# ============================================================================

# Create default config instance
_default_config = Config()

# Export constants
BASE_MODEL = _default_config.model_name
CONNECTOR_ATTENTION_WEIGHT = _default_config.connector_attention_weight
DISCOURSE_ATTENTION_SCALING = _default_config.discourse_attention_scaling
MAX_SEQUENCE_LENGTH = _default_config.max_sequence_length

# Tag format (YOUR EXACT FORMAT)
TAG_FORMAT = '<connector type="{type}"> {word} </connector>'

# Dataset configuration
DATASETS = {
    'arxiv': {
        'name': 'armanc/scientific_papers',
        'config': 'arxiv',
        'split': 'train',
        'description': 'Scientific papers from ArXiv'
    },
    'pubmed': {
        'name': 'armanc/scientific_papers',
        'config': 'pubmed',
        'split': 'train',
        'description': 'Biomedical papers from PubMed'
    },
    'legal': {
        'sources': [
            ('pile-of-law/pile-of-law', 'r_legaladvice'),
            ('jonathanli/pile-of-law-sample', None)
        ],
        'description': 'Legal documents and case law'
    },
    'openwebmath': {
        'name': 'open-web-math/open-web-math',
        'config': None,
        'split': 'train',
        'description': 'High-quality mathematical text'
    }
}

COMBINED_DATASET_PATH = _default_config.combined_dataset_path
DATASET_NAME = "armanc/scientific_papers"  # Legacy
DATASET_CONFIG = "arxiv"  # Legacy
DATASET_SPLIT = "train"  # Legacy

# Preprocessing
CHECKPOINT_SIZE = _default_config.checkpoint_size
MIN_PAPER_LENGTH = _default_config.min_paper_length
MAX_PAPER_LENGTH = _default_config.max_paper_length
CHECKPOINT_DIR = _default_config.checkpoint_dir

# Special tokens (YOUR FORMAT)
SPECIAL_TOKENS = _default_config.get_special_tokens()

# ============================================================================
# CONNECTOR PATTERNS (COMPREHENSIVE REGEX)
# ============================================================================

CONNECTOR_PATTERNS = {
    'causal': [
        r'\bbecause\b', r'\bsince\b', r'\bas\b', r'\bfor\b',
        r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bconsequently\b',
        r'\baccordingly\b', r'\bas a result\b', r'\bas a consequence\b',
        r'\bfor this reason\b', r'\bthat is why\b', r'\bso\b',
        r'\bthen\b', r'\bthereby\b', r'\bwherefore\b', r'\bergo\b',
        r'\bgiven that\b', r'\bseeing that\b', r'\bin that\b',
        r'\binasmuch as\b', r'\binsofar as\b', r'\bon the grounds that\b',
        r'\bdue to\b', r'\bowing to\b', r'\bon account of\b',
        r'\bby virtue of\b', r'\bby reason of\b', r'\bthanks to\b',
        r'\bso that\b', r'\bin order that\b', r'\bin order to\b',
        r'\bso as to\b', r'\bwith the aim of\b', r'\bfor the purpose of\b',
        r'\bwith a view to\b', r'\bto the end that\b'
    ],
    'adversative': [
        r'\bbut\b', r'\bhowever\b', r'\byet\b', r'\bwhereas\b',
        r'\bwhile\b', r'\bwhilst\b', r'\bon the contrary\b',
        r'\bin contrast\b', r'\bconversely\b', r'\bby contrast\b',
        r'\bon the other hand\b', r'\bin opposition to\b',
        r'\bas opposed to\b', r'\brather than\b', r'\bunlike\b',
        r'\balthough\b', r'\bthough\b', r'\beven though\b',
        r'\beven if\b', r'\bdespite\b', r'\bin spite of\b',
        r'\bnotwithstanding\b', r'\bregardless of\b', r'\badmittedly\b',
        r'\bgranted that\b', r'\bgranting that\b', r'\bbe that as it may\b',
        r'\ball the same\b', r'\bthat said\b', r'\binstead\b',
        r'\brather\b', r'\balternatively\b', r'\bon the flip side\b',
        r'\bthen again\b', r'\botherwise\b', r'\bor else\b',
        r'\bif not\b', r'\bnevertheless\b', r'\bnonetheless\b',
        r'\bstill\b', r'\banyway\b', r'\banyhow\b',
        r'\bat any rate\b', r'\bin any case\b', r'\bin any event\b',
        r'\bafter all\b', r'\bat the same time\b'
    ],
    'temporal': [
        r'\bwhen\b', r'\bwhile\b', r'\bas\b', r'\bwhenever\b',
        r'\bas long as\b', r'\bat the same time\b', r'\bsimultaneously\b',
        r'\bmeantime\b', r'\bmeanwhile\b', r'\bin the meantime\b',
        r'\bat that moment\b', r'\bat that time\b', r'\bjust then\b',
        r'\bbefore\b', r'\bprior to\b', r'\bpreviously\b',
        r'\bearlier\b', r'\bformerly\b', r'\bbeforehand\b',
        r'\bahead of\b', r'\bin advance of\b', r'\buntil\b',
        r'\btill\b', r'\bup to\b', r'\bup until\b',
        r'\bafter\b', r'\bafterwards\b', r'\bafterward\b',
        r'\bfollowing\b', r'\bsubsequently\b', r'\blater\b',
        r'\bnext\b', r'\bthen\b', r'\bsince\b', r'\bfrom then on\b',
        r'\bthenceforth\b', r'\bthereafter\b', r'\bonce\b',
        r'\bas soon as\b', r'\bimmediately after\b',
        r'\bfirst\b', r'\bfirstly\b', r'\bsecond\b', r'\bsecondly\b',
        r'\bthird\b', r'\bthirdly\b', r'\bfinally\b', r'\blastly\b',
        r'\bin the end\b', r'\beventually\b', r'\bultimately\b',
        r'\bto begin with\b', r'\bto start with\b', r'\binitially\b',
        r'\bduring\b', r'\bthroughout\b'
    ],
    'conditional': [
        r'\bif\b', r'\bwhen\b', r'\bwhenever\b', r'\bonce\b',
        r'\bas soon as\b', r'\bthe moment\b', r'\bsuppose\b',
        r'\bsupposing\b', r'\bassuming\b', r'\bprovided that\b',
        r'\bproviding that\b', r'\bgiven that\b', r'\bgranted that\b',
        r'\bon condition that\b', r'\bin the event that\b',
        r'\bin case\b', r'\bjust in case\b', r'\bunless\b',
        r'\bexcept if\b', r'\bif not\b', r'\bonly if\b',
        r'\bbut for\b', r'\bsave that\b', r'\bwere it not for\b',
        r'\bif it were not for\b'
    ],
    'conclusive': [
        r'\btherefore\b', r'\bthus\b', r'\bhence\b', r'\bso\b',
        r'\bthen\b', r'\bit follows that\b', r'\bfrom this it follows\b',
        r'\bthe conclusion is\b', r'\bwe can conclude that\b',
        r'\bin conclusion\b', r'\bto conclude\b', r'\bin summary\b',
        r'\bto summarize\b', r'\bto sum up\b', r'\bin sum\b',
        r'\bin short\b', r'\bin brief\b', r'\ball in all\b',
        r'\bon the whole\b', r'\boverall\b', r'\bin overview\b',
        r'\bin general\b', r'\bgenerally speaking\b',
        r'\bin other words\b', r'\bthat is\b', r'\bthat is to say\b',
        r'\bnamely\b', r'\bspecifically\b', r'\bto put it another way\b',
        r'\bto rephrase\b', r'\bi\.e\.\b', r'\bviz\.\b',
        r'\bfinally\b', r'\blastly\b', r'\blast but not least\b',
        r'\bin the final analysis\b', r'\bat last\b',
        r'\bultimately\b', r'\bin the end\b'
    ],
    'additive': [
        r'\band\b', r'\balso\b', r'\btoo\b', r'\bas well\b',
        r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
        r'\badditionally\b', r'\bbesides\b', r'\bwhat is more\b',
        r'\bon top of that\b', r'\bplus\b', r'\balong with\b',
        r'\btogether with\b', r'\bcoupled with\b', r'\bnot to mention\b',
        r'\blikewise\b', r'\bsimilarly\b', r'\bequally\b',
        r'\bin the same way\b', r'\bby the same token\b',
        r'\bin like manner\b', r'\bcorrespondingly\b', r'\banalogously\b',
        r'\bindeed\b', r'\bin fact\b', r'\bactually\b',
        r'\bas a matter of fact\b', r'\bin truth\b', r'\bto tell the truth\b',
        r'\bfrankly\b', r'\bclearly\b', r'\bobviously\b',
        r'\bevidently\b', r'\bnotably\b', r'\bsignificantly\b',
        r'\bimportantly\b', r'\bfor example\b', r'\bfor instance\b',
        r'\bsuch as\b', r'\bnamely\b', r'\be\.g\.\b',
        r'\bto illustrate\b', r'\bto demonstrate\b',
        r'\bas an illustration\b', r'\bin particular\b',
        r'\bparticularly\b', r'\bespecially\b'
    ]
}

# Logging
LOG_LEVEL = _default_config.log_level
SAVE_TEXT_PREVIEW = _default_config.save_text_preview
PREVIEW_LENGTH = _default_config.preview_length

# Performance
NUM_WORKERS = _default_config.num_workers
BATCH_SIZE = _default_config.batch_size


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_connector_tag(connector_type: str, word: str) -> str:
    """
    Format connector tag in your exact format.
    
    Args:
        connector_type: Type (causal, adversative, etc.)
        word: The connector word
    
    Returns:
        <connector type="X"> word </connector>
    """
    return TAG_FORMAT.format(type=connector_type, word=word)


def get_connector_types() -> List[str]:
    """Get connector type names (uppercase)."""
    return _default_config.get_connector_type_names()


def get_dataset_info(dataset_name: str) -> Optional[Dict]:
    """Get information about a specific dataset."""
    return DATASETS.get(dataset_name.lower())


def get_all_datasets() -> List[str]:
    """Get list of all supported dataset names."""
    return list(DATASETS.keys())


def verify_configuration():
    """Verify all configuration settings."""
    print("\n" + "="*70)
    print("DISCOURSE-AWARE CONFIGURATION VERIFICATION")
    print("="*70)
    
    config = _default_config
    config.validate()
    
    print(f"✓ Model: {config.model_name}")
    print(f"✓ Tag format: {TAG_FORMAT}")
    print(f"✓ Attention scaling: {config.discourse_attention_scaling}x (pre-softmax)")
    print(f"✓ Loss weighting: {config.connector_attention_weight}x")
    print(f"✓ LoRA rank: {config.lora_r}")
    print(f"✓ Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"✓ Connector types: {len(config.connector_types)}")
    print(f"✓ Datasets: {len(DATASETS)}")
    
    print("\nConnector types:")
    for ctype in config.connector_types.keys():
        patterns = len(CONNECTOR_PATTERNS.get(ctype, []))
        print(f"  - {ctype.upper()}: {patterns} regex patterns")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================

if __name__ == "__main__":
    verify_configuration()