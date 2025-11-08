#!/usr/bin/env python3
"""
model_validated_final.py - UPDATED TO HANDLE MULTI-WORD CONNECTORS

CRITICAL UPDATE:
1. Detects opening and closing connector tags
2. Identifies ALL tokens between opening and closing tags
3. Boosts ALL enclosed tokens (handles "in general", "in conclusion", etc.)
4. Works with single-word AND multi-word connector phrases
5. Applies boost ONCE at input (no per-layer compounding)

EXAMPLES:
✓ <connector type="CAUSAL">because</connector>
✓ <connector type="CONCLUSIVE">in conclusion</connector>  (multi-word)
✓ <connector type="ADDITIVE">for example</connector>  (multi-word)

All enclosed tokens get 1.1× boost!
"""

import torch
import torch.nn as nn
import logging
import types
from pathlib import Path
from typing import Optional, Dict, Set
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
    """Transformer block - STANDARD (no boost here)."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.ff = FeedForward(config)
        
        self.norm_attn = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.norm_ff = RMSNorm(config["emb_dim"], config["norm_eps"])
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard forward pass - no connector boosting."""
        
        # Attention block with residual
        normed = self.norm_attn(x)
        attn_out = self.attn(normed, mask, cos, sin)
        x = x + attn_out
        
        # Feed forward block with residual
        normed = self.norm_ff(x)
        ff_out = self.ff(normed)
        x = x + ff_out
        
        return x


class Llama3Model(nn.Module):
    """
    Llama 3.2 with MULTI-WORD connector support.
    
    UPDATED TO HANDLE:
    - Single-word: <connector>because</connector>
    - Multi-word: <connector>in conclusion</connector>
    - Boosts ALL enclosed tokens equally
    """
    
    def __init__(self, config: dict, tokenizer, boost_factor: float = 1.1):
        super().__init__()
        
        self.config_dict = types.SimpleNamespace(**config)
        self.boost_factor = boost_factor
        self.tokenizer = tokenizer
        
        # Token IDs for connector tags
        self.connector_opening_tags: Set[int] = set()
        self.connector_closing_tag_id: Optional[int] = None
        
        # Extract connector tag token IDs
        self._setup_connector_tags()
        
        self.token_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        
        self.rope_emb = RoPEPositionalEmbedding(
            dim=config["emb_dim"] // config["n_heads"],
            max_seq_len=config["context_length"],
            rope_base=config["rope_base"]
        )
        
        # Standard transformer blocks (no boost)
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config["n_layers"])
        ])
        
        self.norm_final = RMSNorm(config["emb_dim"], config["norm_eps"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def _setup_connector_tags(self):
        """Extract connector tag token IDs from tokenizer."""
        logger.info("Extracting connector tag token IDs...")
        
        # Opening tags
        opening_tag_templates = [
            '<connector type="CAUSAL">',
            '<connector type="ADVERSATIVE">',
            '<connector type="TEMPORAL">',
            '<connector type="CONDITIONAL">',
            '<connector type="CONCLUSIVE">',
            '<connector type="ADDITIVE">',
        ]
        
        for tag_str in opening_tag_templates:
            try:
                token_id = self.tokenizer.encode(tag_str, add_special_tokens=False)
                if len(token_id) == 1:
                    self.connector_opening_tags.add(token_id[0])
                    logger.debug(f"  Found: {tag_str} → ID {token_id[0]}")
            except:
                logger.warning(f"  Could not find: {tag_str}")
        
        # Closing tag
        closing_tag_str = '</connector>'
        try:
            token_id = self.tokenizer.encode(closing_tag_str, add_special_tokens=False)
            if len(token_id) == 1:
                self.connector_closing_tag_id = token_id[0]
                logger.debug(f"  Found: {closing_tag_str} → ID {token_id[0]}")
        except:
            logger.warning(f"  Could not find: {closing_tag_str}")
        
        logger.info(f"✓ Found {len(self.connector_opening_tags)} opening tags")
        logger.info(f"✓ Found closing tag: {self.connector_closing_tag_id}")

    def _create_boost_mask(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Create boost mask by detecting connector tag boundaries.
        
        UPDATED ALGORITHM:
        1. Scan for opening connector tags
        2. Mark ALL enclosed tokens until closing tag found
        3. Support multi-word connector phrases
        
        Returns:
            boost_mask: [batch, seq_len] with values 1.0 or 1.1
        """
        batch_size, seq_len = in_idx.shape
        device = in_idx.device
        
        # Initialize mask (all 1.0)
        boost_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
        
        # Process each sequence in batch
        for batch_idx in range(batch_size):
            sequence = in_idx[batch_idx]
            
            i = 0
            while i < seq_len:
                token_id = sequence[i].item()
                
                # Found an opening connector tag
                if token_id in self.connector_opening_tags:
                    logger.debug(f"Found opening tag at position {i}")
                    
                    # Scan forward to find closing tag
                    closing_pos = None
                    for j in range(i + 1, seq_len):
                        if sequence[j].item() == self.connector_closing_tag_id:
                            closing_pos = j
                            break
                    
                    if closing_pos is not None:
                        # BOOST ALL TOKENS BETWEEN opening and closing tags
                        # This handles multi-word connectors!
                        for pos in range(i + 1, closing_pos):
                            boost_mask[batch_idx, pos] = self.boost_factor
                        
                        num_boosted = closing_pos - i - 1
                        logger.debug(
                            f"  Boosted {num_boosted} tokens (positions {i+1} to {closing_pos-1})"
                        )
                        
                        # Skip past closing tag
                        i = closing_pos + 1
                    else:
                        logger.warning(f"  No closing tag found after position {i}")
                        i += 1
                else:
                    i += 1
        
        return boost_mask

    @property
    def config(self):
        """Return config for HuggingFace."""
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

    def forward(
        self,
        in_idx: torch.Tensor,
        connector_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multi-word connector support.
        
        HANDLES:
        ✓ Single-word: "because"
        ✓ Multi-word: "in general", "in conclusion", "for example"
        ✓ Boosts ALL enclosed tokens equally
        
        Args:
            in_idx: [batch, seq_len] token IDs
            connector_mask: [OPTIONAL] If provided, use this
        
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = in_idx.shape
        
        # Create or use provided boost mask
        if connector_mask is not None:
            boost_mask = connector_mask.unsqueeze(-1)
        else:
            # Auto-detect connector tag boundaries (MULTI-WORD SUPPORT)
            boost_mask_2d = self._create_boost_mask(in_idx)
            boost_mask = boost_mask_2d.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Embed tokens
        x = self.token_emb(in_idx)  # [batch, seq_len, emb_dim]
        
        # BOOST CONNECTOR WORDS AT INPUT (MULTI-WORD SUPPORT)
        x = x * boost_mask
        
        # Generate position embeddings
        cos, sin = self.rope_emb(seq_len, in_idx.device)
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=in_idx.device, dtype=torch.bool),
            diagonal=1
        ).unsqueeze(0)
        mask = ~mask
        
        # Process through all transformer blocks (no boost)
        for block in self.trf_blocks:
            x = block(x, mask, cos, sin)
        
        # Final normalization
        x = self.norm_final(x)
        
        # Output projection
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
        
        self.boost_factor = config.boost_factor
        
        if "3B" in config.model_name:
            self.model_size = "3B"
        elif "1B" in config.model_name:
            self.model_size = "1B"
        else:
            self.model_size = "1B"
        
        logger.info(f"Model handler initialized")
        logger.info(f"  Size: {self.model_size}")
        logger.info(f"  Boost: {self.boost_factor}x (multi-word support)")
    
    @property
    def config(self):
        """Return config for Trainer."""
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
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed: {e}")
            raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"✓ Vocab: {self.original_vocab_size:,}")
        
        return self.tokenizer
    
    def extend_tokenizer(self, special_tokens):
        """Add special tokens."""
        if not special_tokens:
            return 0
        
        logger.info(f"Adding {len(special_tokens)} tokens...")
        num_added = self.tokenizer.add_tokens(special_tokens)
        
        logger.info(f"✓ Added {num_added}, new vocab: {len(self.tokenizer):,}")
        return num_added
    
    def load_model(self):
        """Load model."""
        logger.info(f"Loading Llama 3.2 ({self.model_size})")
        logger.info(f"  Multi-word connector support: ENABLED")
        
        cfg = get_model_config(self.model_size)
        cfg["vocab_size"] = len(self.tokenizer)
        
        self.model = Llama3Model(cfg, tokenizer=self.tokenizer, boost_factor=self.boost_factor)
        
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        logger.info(f"✓ On {device}")
        
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model info."""
        if self.model is None:
            return {}
        
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": f"Llama 3.2 ({self.model_size})",
            "total_parameters": total,
            "trainable_percentage": 100 * trainable / total if total > 0 else 0,
            "boost_factor": self.boost_factor,
            "multi_word_support": "YES"
        }
    
    def print_model_info(self):
        """Print model info."""
        info = self.get_model_info()
        
        logger.info("\n" + "="*70)
        logger.info("MODEL INFORMATION")
        logger.info("="*70)
        logger.info(f"Model: {info.get('model_type', 'N/A')}")
        logger.info(f"Parameters: {info.get('total_parameters', 0):,}")
        logger.info(f"Trainable: {info.get('trainable_percentage', 0):.1f}%")
        logger.info(f"\nConnector Boosting:")
        logger.info(f"  Factor: {info.get('boost_factor', 1.0)}x")
        logger.info(f"  Multi-word support: {info.get('multi_word_support', 'N/A')}")
        logger.info(f"  Applied: At input (all enclosed tokens)")
        logger.info(f"  Gradient amp: 1.1× only (not 1.1^28)")
        logger.info("="*70 + "\n")
    
    def save_model(self, output_path: str):
        """Save model."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to {output_dir}")
        
        try:
            torch.save(self.model.state_dict(), output_dir / "model.pt")
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.model.config, output_dir / "model_config.pt")
            logger.info("✓ Saved")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise
    
    def load_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Not found: {checkpoint_dir}")
            return False
        
        try:
            device = torch.device(self.config.device)
            state_dict = torch.load(checkpoint_dir / "model.pt", map_location=device)
            self.model.load_state_dict(state_dict)
            logger.info("✓ Loaded")
            return True
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False


def initialize_model(config) -> Model:
    """Initialize model."""
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING LLAMA 3.2 (MULTI-WORD SUPPORT)")
    logger.info("="*70)
    
    model_handler = Model(config)
    
    logger.info("\n[1/3] Tokenizer...")
    model_handler.load_tokenizer()
    
    logger.info("\n[2/3] Extending tokenizer...")
    model_handler.extend_tokenizer(config.get_special_tokens())
    
    logger.info("\n[3/3] Model...")
    model_handler.load_model()
    
    model_handler.print_model_info()
    
    return model_handler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from utils.config import Config
        config = Config()
        model_handler = initialize_model(config)
        logger.info("✓ Success!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
