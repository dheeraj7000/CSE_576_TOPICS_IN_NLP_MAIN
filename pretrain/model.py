#!/usr/bin/env python3
"""
model.py - COMPLETE INTEGRATED VERSION (WITH CONFIG.PY)

Llama 3.2 with connector-aware hidden state boosting.
Includes model architecture + handler for loading, saving, checkpointing.

USES: config.py for all configuration (model name, device, boost_factor, etc.)

ARCHITECTURAL CHANGES:
1. Connector boosting applied ONCE per TransformerBlock (at the end)
2. NO compounding (1.1 only, not 1.1²)
3. Amplifies gradients for connector positions
4. Improves weight updates for logical connectors
5. Enables better reasoning learning during pretraining

Reference: https://sebastianraschka.com/llms-from-scratch/ch05/07_gpt_to_llama/
"""

import torch
import torch.nn as nn
import logging
import types
from pathlib import Path
from typing import Optional, Dict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def get_model_config(size: str = "1B") -> dict:
    """Get Llama 3.2 config."""
    configs = {
        "1B": {
            "vocab_size": 128256,
            "context_length": 2048,
            "emb_dim": 2048,
            "n_heads": 32,
            "n_layers": 16,
            "hidden_dim": 8192,
            "n_kv_heads": 8,
            "rope_base": 500000.0,
            "rope_freq_config_str": "default",
            "norm_eps": 1e-5,
        },
        "3B": {
            "vocab_size": 128256,
            "context_length": 2048,
            "emb_dim": 3072,
            "n_heads": 24,
            "n_layers": 28,
            "hidden_dim": 8192,
            "n_kv_heads": 8,
            "rope_base": 500000.0,
            "rope_freq_config_str": "default",
            "norm_eps": 1e-5,
        }
    }
    return configs[size]


class RoPEPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int, rope_base: float):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seq_len: int, device: torch.device):
        """Generate RoPE embeddings for sequence."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor."""
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    
    x_rot = torch.cat([-x[..., head_dim//2:], x[..., :head_dim//2]], dim=-1)
    
    x_rope = x * cos + x_rot * sin
    
    return x_rope


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.n_kv_heads = config["n_kv_heads"]
        self.head_dim = config["emb_dim"] // config["n_heads"]
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        self.w_q = nn.Linear(config["emb_dim"], config["emb_dim"], bias=False)
        self.w_k = nn.Linear(config["emb_dim"], config["n_kv_heads"] * self.head_dim, bias=False)
        self.w_v = nn.Linear(config["emb_dim"], config["n_kv_heads"] * self.head_dim, bias=False)
        self.w_o = nn.Linear(config["emb_dim"], config["emb_dim"], bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, emb_dim = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        
        rep = self.n_heads // self.n_kv_heads
        k = k.unsqueeze(3).expand(batch_size, seq_len, self.n_kv_heads, rep, self.head_dim)
        v = v.unsqueeze(3).expand(batch_size, seq_len, self.n_kv_heads, rep, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_2d == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, emb_dim)
        
        output = self.w_o(context)
        
        return output


class FeedForward(nn.Module):
    """SiLU-gated Feed Forward Network."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.up = nn.Linear(config["emb_dim"], config["hidden_dim"], bias=False)
        self.gate = nn.Linear(config["emb_dim"], config["hidden_dim"], bias=False)
        self.down = nn.Linear(config["hidden_dim"], config["emb_dim"], bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / norm) * self.weight


class TransformerBlock(nn.Module):
    """
    Transformer block with connector-aware hidden state boosting.
    
    CORRECTED ARCHITECTURE:
    1. Attention + residual
    2. Feed forward + residual
    3. Connector boost applied ONCE (at the end)
    
    This prevents compounding (1.1 only, not 1.1²).
    Amplifies gradients for connector positions.
    """
    
    def __init__(self, config: dict, boost_factor: float = 1.1):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ff = FeedForward(config)
        
        self.norm_attn = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.norm_ff = RMSNorm(config["emb_dim"], config["norm_eps"])
        
        self.boost_factor = boost_factor
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        connector_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        CORRECTED forward pass with connector boosting.
        
        Flow:
            x → norm → attention → residual → x₁
            x₁ → norm → feed_forward → residual → x₂
            x₂ → connector_boost → output (ONCE, no compounding)
        """
        # Attention block with residual
        normed = self.norm_attn(x)
        attn_out = self.attn(normed, mask, cos, sin)
        x = x + attn_out
        
        # Feed forward block with residual
        normed = self.norm_ff(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        # CONNECTOR BOOST: Applied ONCE at the end (no compounding)
        if connector_mask is not None:
            boost = connector_mask.unsqueeze(-1)
            x = x * boost
        
        return x

class Llama3Model(nn.Module):
    """Llama 3.2 with connector-aware hidden state boosting."""
    
    def __init__(self, config: dict, boost_factor: float = 1.1):
        super().__init__()
        
        # Store config as SimpleNamespace instead of dict
        self.config_dict = types.SimpleNamespace(**config)
        self.boost_factor = boost_factor
        
        # Rest of initialization
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        
        self.rope_emb = RoPEPositionalEmbedding(
            dim=config["emb_dim"] // config["n_heads"],
            max_seq_len=config["context_length"],
            rope_base=config["rope_base"]
        )
        
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config, boost_factor=boost_factor)
            for _ in range(config["n_layers"])
        ])
        
        self.norm_final = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    @property
    def config(self):
        """Return config object for HuggingFace compatibility."""
        
        # Create a proper config wrapper class (not stored as instance attributes)
        class ConfigWrapper:
            """Wrapper that provides config values AND HF-required methods."""
            
            def __init__(self, config_ns):
                self._config_ns = config_ns
                # Add required special token IDs
                self._config_ns.eos_token_id = 128009
                self._config_ns.bos_token_id = 128000
                self._config_ns.pad_token_id = 128009
                self._config_ns.is_encoder_decoder = False
                self._config_ns.use_cache = False
            
            def __getattr__(self, key):
                """Delegate attribute access to wrapped config."""
                return getattr(self._config_ns, key)
            
            def to_json_string(self):
                """Return JSON string - only config data, NO functions."""
                import json
                # Filter out any non-serializable attributes
                cfg_dict = {}
                for key, value in vars(self._config_ns).items():
                    # Only include serializable types
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cfg_dict[key] = value
                return json.dumps(cfg_dict)
            
            def to_dict(self):
                """Return config as dict - only data, NO functions."""
                cfg_dict = {}
                for key, value in vars(self._config_ns).items():
                    # Only include serializable types
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cfg_dict[key] = value
                return cfg_dict
        
        return ConfigWrapper(self.config_dict)

    def forward(
        self,
        in_idx: torch.Tensor,
        connector_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with connector-aware boosting.
        
        Args:
            in_idx: [batch, seq_len] token IDs
            connector_mask: [batch, seq_len] boost mask (1.0 or 1.1)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        
        x = self.token_emb(in_idx)
        
        cos, sin = self.rope_emb(seq_len, in_idx.device)
        
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=in_idx.device, dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0)
        mask = ~mask
        
        for block in self.trf_blocks:
            x = block(x, mask, cos, sin, connector_mask)
        
        x = self.norm_final(x)
        
        logits = self.out_head(x)
        
        return logits

# ============================================================================
# MODEL HANDLER (USES CONFIG.PY)
# ============================================================================

class Model:
    """
    Model handler for connector-boosted Llama 3.2.
    
    Imports from config.py for all settings:
    - model_name: "meta-llama/Llama-3.2-3B"
    - device: "cuda"
    - boost_factor: 1.1
    - Special tokens from get_special_tokens()
    
    Manages:
    - Tokenizer loading and extension
    - Model initialization with connector boosting
    - Device management
    - Checkpointing and loading
    - Model information
    """

    def __init__(self, config):
        """Initialize from config.py"""
        self.config_dict = config
        self.tokenizer = None
        self.model = None
        self.original_vocab_size = None
        
        # From config.py
        self.boost_factor = config.boost_factor
        
        # Determine model size from model_name
        if "3B" in config.model_name:
            self.model_size = "3B"
        elif "1B" in config.model_name:
            self.model_size = "1B"
        else:
            self.model_size = "1B"  # Default
        
        logger.info(f"Initialized Model handler")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Model size: {self.model_size}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Boost factor: {self.boost_factor}")
    
    @property
    def config(self):
        """Return config object for HuggingFace Trainer compatibility."""
        class LlamaConfig:
            def __init__(self, cfg_dict):
                for key, value in cfg_dict.items():
                    setattr(self, key, value)
                # Llama 3.2 special tokens
                self.eos_token_id = 128009
                self.bos_token_id = 128000
                self.pad_token_id = 128009
                self.is_encoder_decoder = False
                self.use_cache = False
        
        # CORRECTED: Handle both dict and Config objects
        if isinstance(self.config_dict, dict):
            cfg_dict = self.config_dict
        else:
            # If it's a Config object, convert to dict
            cfg_dict = self.config_dict.to_dict()
        
        return LlamaConfig(cfg_dict)


    def load_tokenizer(self):
        """Load tokenizer from config.model_name."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token")
        
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"✓ Original vocab size: {self.original_vocab_size:,}")
        
        return self.tokenizer
    
    def extend_tokenizer(self, special_tokens):
        """Add connector and other special tokens from config.py."""
        if not special_tokens:
            return 0
        
        logger.info(f"Adding {len(special_tokens)} special tokens...")
        
        num_added = self.tokenizer.add_tokens(special_tokens)
        
        logger.info(f"✓ Added {num_added} tokens")
        logger.info(f"✓ New vocab size: {len(self.tokenizer):,}")
        
        return num_added
    
    def load_model(self):
        """
        Load Llama 3.2 model with connector boosting.
        
        Uses model_size determined from config.model_name.
        """
        logger.info(f"Loading Llama 3.2 ({self.model_size}) with connector boosting")
        
        # Get model config
        cfg = get_model_config(self.model_size)
        
        # Update vocab size to match extended tokenizer
        cfg["vocab_size"] = len(self.tokenizer)
        logger.info(f"  Updated vocab_size: {cfg['vocab_size']:,}")
        
        # Create model with connector boosting from config
        self.model = Llama3Model(cfg, boost_factor=self.boost_factor)
        
        # Move to device from config
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        logger.info(f"  ✓ Model moved to device: {device}")
        
        logger.info(f"✓ Model loaded successfully")
        
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model statistics and information."""
        if self.model is None:
            logger.warning("Model not loaded")
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_pct = 100 * trainable_params / total_params if total_params > 0 else 0
        
        device = str(next(self.model.parameters()).device)
        dtype = str(next(self.model.parameters()).dtype)
        
        info = {
            "model_name": self.config.model_name,
            "model_type": f"Llama 3.2 ({self.model_size})",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": trainable_pct,
            "original_vocab_size": self.original_vocab_size,
            "current_vocab_size": len(self.tokenizer),
            "device": device,
            "dtype": dtype,
            "connector_boosting": True,
            "boost_factor": self.boost_factor
        }
        
        return info
    
    def print_model_info(self):
        """Print comprehensive model information."""
        info = self.get_model_info()
        
        logger.info("\n" + "="*70)
        logger.info("MODEL INFORMATION")
        logger.info("="*70)
        logger.info(f"Model: {info.get('model_type', 'N/A')}")
        logger.info(f"Total parameters: {info.get('total_parameters', 0):,}")
        logger.info(f"Trainable: {info.get('trainable_parameters', 0):,} ({info.get('trainable_percentage', 0):.1f}%)")
        logger.info(f"Vocab: {info.get('original_vocab_size', 0):,} → {info.get('current_vocab_size', 0):,}")
        logger.info(f"Device: {info.get('device', 'N/A')}")
        logger.info(f"Dtype: {info.get('dtype', 'N/A')}")
        logger.info(f"Connector boosting: {info.get('connector_boosting', False)} ({info.get('boost_factor', 1.0)}x)")
        logger.info("="*70 + "\n")
    
    def save_model(self, output_path: str):
        """Save model state dict and tokenizer."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        try:
            # Save model state dict
            model_path = output_dir / "model.pt"
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"✓ Model state dict saved")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"✓ Tokenizer saved")
            
            # Save config
            config_path = output_dir / "model_config.pt"
            torch.save(self.model.config, config_path)
            logger.info(f"✓ Model config saved")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def load_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model and tokenizer from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False
        
        logger.info(f"Loading from checkpoint: {checkpoint_dir}")
        
        try:
            # Load model state dict
            model_path = checkpoint_dir / "model.pt"
            if model_path.exists():
                device = torch.device(self.config.device)
                state_dict = torch.load(model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✓ Model loaded")
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                logger.info(f"✓ Tokenizer loaded")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}", exc_info=True)
            return False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        logger.info("Enabling gradient checkpointing...")
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("✓ Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not available")


def initialize_model(config) -> Model:
    """
    Initialize model with full setup from config.py.
    
    Usage:
        from config import Config
        config = Config()
        model_handler = initialize_model(config)
        
        model = model_handler.model
        tokenizer = model_handler.tokenizer
    """
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING LLAMA 3.2 MODEL (FROM CONFIG.PY)")
    logger.info("="*70)
    
    # Create handler with config.py
    model_handler = Model(config)
    
    # Load tokenizer
    logger.info("\n[1/3] Loading tokenizer...")
    model_handler.load_tokenizer()
    
    # Extend tokenizer with special tokens from config.py
    logger.info("\n[2/3] Extending tokenizer...")
    special_tokens = config.get_special_tokens()
    model_handler.extend_tokenizer(special_tokens)
    
    # Load model
    logger.info("\n[3/3] Loading model...")
    model_handler.load_model()
    
    # Print info
    model_handler.print_model_info()
    
    return model_handler


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test with config.py
    try:
        from utils.config import Config
        
        logger.info("Loading config from config.py...")
        config = Config()
        config.print_summary()
        
        logger.info("Initializing model...")
        model_handler = initialize_model(config)
        
        logger.info("✓ Test successful!")
        
    except ImportError:
        logger.error("Please ensure config.py is in the same directory")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)