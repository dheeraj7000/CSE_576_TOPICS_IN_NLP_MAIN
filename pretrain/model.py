#!/usr/bin/env python3
"""
model.py - CLEAN FINAL VERSION

Llama 3.2 with connector-aware training.

KEY ARCHITECTURAL CHANGES:
✅ NO connector_mask in forward() parameters
✅ TransformerBlock ONLY takes: x, mask, cos, sin
✅ No boost logic in TransformerBlock (data_loader handles it)
✅ Simple, clean transformer implementation
"""

import torch
import torch.nn as nn
import logging
import types
from pathlib import Path
from typing import Optional, Dict
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def get_model_config(size: str = "3B") -> dict:
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
    cos = cos[:seq_len, :head_dim]
    sin = sin[:seq_len, :head_dim]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                cos: Optional[torch.Tensor] = None, sin: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            causal_mask = ~causal_mask
            combined_mask = mask_2d & causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~combined_mask, float('-inf'))
        
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
    Transformer block - CLEAN VERSION.
    
    ✅ NO connector_mask parameter
    ✅ Simple transformer: attention → residual → FF → residual
    ✅ No boost logic here (data_loader handles it)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ff = FeedForward(config)
        self.norm_attn = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.norm_ff = RMSNorm(config["emb_dim"], config["norm_eps"])
    
    def forward(self, x, mask, cos, sin):
        """Forward pass - NO connector_mask!"""
        # Attention + residual
        normed = self.norm_attn(x)
        attn_out = self.attn(normed, mask, cos, sin)
        x = x + attn_out
        
        # Feed forward + residual
        normed = self.norm_ff(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


class Llama3Model(nn.Module):
    """Llama 3.2 model - clean implementation."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config_dict = types.SimpleNamespace(**config)
        
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.rope_emb = RoPEPositionalEmbedding(
            dim=config["emb_dim"] // config["n_heads"],
            max_seq_len=config["context_length"],
            rope_base=config["rope_base"]
        )
        
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config["n_layers"])
        ])
        
        self.norm_final = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
    
    @property
    def config(self):
        """Return config object for HuggingFace compatibility."""
        class ConfigWrapper:
            def __init__(self, config_ns):
                self._config_ns = config_ns
                self._config_ns.eos_token_id = 128009
                self._config_ns.bos_token_id = 128000
                self._config_ns.pad_token_id = 128009
                self._config_ns.is_encoder_decoder = False
                self._config_ns.use_cache = False
            
            def __getattr__(self, key):
                return getattr(self._config_ns, key)
            
            def to_json_string(self):
                import json
                cfg_dict = {}
                for key, value in vars(self._config_ns).items():
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cfg_dict[key] = value
                return json.dumps(cfg_dict)
            
            def to_dict(self):
                cfg_dict = {}
                for key, value in vars(self._config_ns).items():
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cfg_dict[key] = value
                return cfg_dict
        
        return ConfigWrapper(self.config_dict)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - ONLY input_ids!
        
        ✅ NO connector_mask parameter
        ✅ Clean interface for trainer
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
            x = block(x, mask, cos, sin)
        
        x = self.norm_final(x)
        logits = self.out_head(x)
        
        return logits


# ============================================================================
# MODEL HANDLER
# ============================================================================

class Model:
    """Model handler for Llama 3.2."""
    
    def __init__(self, config):
        self.config_dict = config
        self.tokenizer = None
        self.model = None
        self.original_vocab_size = None
        
        if "3B" in config.model_name:
            self.model_size = "3B"
        elif "1B" in config.model_name:
            self.model_size = "1B"
        else:
            self.model_size = "3B"
        
        logger.info(f"✓ Model handler initialized")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Size: {self.model_size}")
        logger.info(f"  Device: {config.device}")
    
    @property
    def config(self):
        """Return config object for HuggingFace compatibility."""
        class LlamaConfig:
            def __init__(self, cfg_dict):
                for key, value in cfg_dict.items():
                    setattr(self, key, value)
                self.eos_token_id = 128009
                self.bos_token_id = 128000
                self.pad_token_id = 128009
                self.is_encoder_decoder = False
                self.use_cache = False
        
        if isinstance(self.config_dict, dict):
            cfg_dict = self.config_dict
        else:
            cfg_dict = self.config_dict.to_dict()
        
        return LlamaConfig(cfg_dict)
    
    def load_tokenizer(self):
        """Load tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"✓ Tokenizer loaded - vocab size: {self.original_vocab_size:,}")
        return self.tokenizer
    
    def extend_tokenizer(self, special_tokens):
        """Add special tokens."""
        if not special_tokens:
            return 0
        
        num_added = self.tokenizer.add_tokens(special_tokens)
        logger.info(f"✓ Added {num_added} special tokens")
        logger.info(f"✓ New vocab size: {len(self.tokenizer):,}")
        return num_added
    
    def load_model(self):
        """Load model."""
        logger.info(f"Loading Llama 3.2 ({self.model_size})")
        
        cfg = get_model_config(self.model_size)
        cfg["vocab_size"] = len(self.tokenizer)
        
        self.model = Llama3Model(cfg)
        
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        
        logger.info(f"✓ Model loaded on {device}")
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model statistics."""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "model_type": f"Llama 3.2 ({self.model_size})",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "original_vocab_size": self.original_vocab_size,
            "current_vocab_size": len(self.tokenizer),
            "device": str(next(self.model.parameters()).device),
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        logger.info("\n" + "="*70)
        logger.info("MODEL INFORMATION")
        logger.info("="*70)
        logger.info(f"Model: {info['model_type']}")
        logger.info(f"Parameters: {info['total_parameters']:,}")
        logger.info(f"Vocab: {info['original_vocab_size']:,} → {info['current_vocab_size']:,}")
        logger.info(f"Device: {info['device']}")
        logger.info("="*70 + "\n")
    
    def save_model(self, output_path: str):
        """Save model and tokenizer."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        torch.save(self.model.state_dict(), output_dir / "model.pt")
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ Model and tokenizer saved")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False
        
        logger.info(f"Loading from checkpoint: {checkpoint_dir}")
        
        device = torch.device(self.config.device)
        state_dict = torch.load(checkpoint_dir / "model.pt", map_location=device)
        self.model.load_state_dict(state_dict)
        logger.info(f"✓ Model loaded")
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        logger.info(f"✓ Tokenizer loaded")
        
        return True


def initialize_model(config) -> Model:
    """Initialize model from config."""
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING LLAMA 3.2 MODEL")
    logger.info("="*70)
    
    model_handler = Model(config)
    
    logger.info("\n[1/3] Loading tokenizer...")
    model_handler.load_tokenizer()
    
    logger.info("\n[2/3] Extending tokenizer...")
    special_tokens = config.get_special_tokens()
    model_handler.extend_tokenizer(special_tokens)
    
    logger.info("\n[3/3] Loading model...")
    model_handler.load_model()
    
    model_handler.print_model_info()
    
    return model_handler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from utils.config import Config
        config = Config()
        config.print_summary()
        
        model_handler = initialize_model(config)
        logger.info("✓ Model initialization successful!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
