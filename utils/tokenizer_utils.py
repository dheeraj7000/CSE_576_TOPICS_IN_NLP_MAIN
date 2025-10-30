#!/usr/bin/env python3
"""
tokenizer_utils.py - CRD CORE: Connector-Aware Tokenization with Mask Generation

Creates connector masks for discourse-aware attention weighting.

KEY TO CRD ARCHITECTURE:
- Tokenizes text with exact tag format: <connector type="X"> word </connector>
- Generates connector_mask: binary mask marking connector positions
- connector_mask values: 0.0 (padding), 1.0 (default), or 1.1 (connectors)
- CRITICAL: This mask is used during training for:
  1. Loss weighting (applied during backward pass)
  2. Attention modification (applied in model forward pass - pre-softmax)

The mask is the BRIDGE between preprocessing and model architecture changes.

IMPORTANT: Padding positions are explicitly set to 0.0 (handled in tokenizer,
not deferred to model). This avoids redundant computation.

Compatible with: config.py, preprocess.py, discourse_training.py, discourse_aware_model.py
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from transformers import PreTrainedTokenizerFast

from config import SPECIAL_TOKENS, CONNECTOR_ATTENTION_WEIGHT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CONNECTOR TOKENIZER - CORE CRD CLASS
# ============================================================================

class ConnectorTokenizer:
    """
    Wrapper for tokenizer with CRD connector-aware mask generation.
    
    CORE FUNCTION: Generates connector_mask for discourse-aware training.
    
    How it works:
    - Input: Text with exact tags like <connector type="CAUSAL"> word </connector>
    - Output: 
      - input_ids: standard token IDs
      - attention_mask: standard mask (1/0)
      - connector_mask: weighted mask (0.0 for padding, 1.0 or 1.1 for tokens)
    
    The connector_mask is used by:
    1. Model attention layer (pre-softmax multiplication)
    2. Loss function (gradient weighting)
    
    This allows model to learn which connector types are important.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        connector_weight: float = CONNECTOR_ATTENTION_WEIGHT
    ):
        """
        Initialize connector tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer (with special tokens added)
            connector_weight: Weight multiplier for connector tokens
                            Default: 1.1 (from config, learnable via training)
        """
        
        self.tokenizer = tokenizer
        self.connector_weight = connector_weight
        self.connector_token_ids = self._get_connector_token_ids()
        
        logger.info(f"✓ ConnectorTokenizer initialized")
        logger.info(f"  Base connector weight: {connector_weight}")
        logger.info(f"  Special tokens found: {len(self.connector_token_ids)}")
    
    def _get_connector_token_ids(self) -> Dict[str, int]:
        """
        Get token IDs for all connector special tokens.
        
        Maps special token strings to their token IDs.
        Used to identify connector tokens in tokenized sequence.
        
        Returns:
            Dict mapping token string to token ID
        """
        
        token_ids = {}
        
        for token in SPECIAL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            
            # Only track tokens successfully added
            if token_id != self.tokenizer.unk_token_id:
                token_ids[token] = token_id
        
        return token_ids
    
    def tokenize_with_connector_mask(
        self,
        text: str,
        max_length: int = 2048,
        return_tensors: Optional[str] = None,
        connector_weight: Optional[float] = None
    ) -> Dict:
        """
        Tokenize text and create CRD connector attention mask.
        
        CRITICAL FOR CRD: This generates the mask that bridges
        preprocessing and model attention modification.
        
        Mask structure:
        - 0.0 for padding tokens (handled here, not in model)
        - 1.0 for regular tokens
        - connector_weight (e.g., 1.1) for tokens inside <connector> tags
        
        This mask is used by:
        1. Loss function: loss *= connector_mask (during training)
        2. Attention module: attention_scores *= connector_mask (pre-softmax)
        
        Args:
            text: Input text with exact tag format
                  Example: "Model works <connector type=\"CAUSAL\"> because </connector> data"
            max_length: Maximum sequence length (default: 2048)
            return_tensors: "pt" for PyTorch tensors, None for lists
            connector_weight: Override weight for this call (default: use instance weight)
        
        Returns:
            Dict containing:
            - 'input_ids': Tokenized IDs (batch_size, seq_len)
            - 'attention_mask': Standard mask (1 for real tokens, 0 for padding)
            - 'connector_mask': Weighted mask (0.0 for padding, 1.0 or connector_weight)
        """
        
        # Use override weight if provided
        weight = connector_weight if connector_weight is not None else self.connector_weight
        
        # Standard tokenization
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None  # Get as lists first
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # CREATE CONNECTOR MASK - CORE OF CRD
        connector_mask = self._create_connector_mask(input_ids, attention_mask, weight)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            connector_mask = torch.tensor(connector_mask, dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'connector_mask': connector_mask
        }
    
    def _create_connector_mask(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        weight: float
    ) -> List[float]:
        """
        Create mask marking connector tokens with specified weight.
        
        CORE LOGIC: Identifies token positions inside <connector> tags
        and marks them with the weight value. Padding positions are
        explicitly set to 0.0 (handled here, not deferred to model).
        
        Logic:
        1. Find opening tag tokens: <connector type="X">
        2. Mark all tokens between opening and closing tags with weight
        3. Mark all padding positions (attention_mask=0) with 0.0
        
        This mask enables:
        - Attention pre-softmax multiplication (architectural)
        - Loss gradient weighting (training)
        
        Args:
            input_ids: List of token IDs from tokenizer
            attention_mask: List of attention mask values (1 for real, 0 for padding)
            weight: Weight to apply to connector tokens
                   (e.g., 1.1 for 10% boost)
        
        Returns:
            List of float mask values (length = len(input_ids))
        """
        
        # Start with attention mask: 1.0 where real, 0.0 where padding
        # This ensures padding is explicitly zeroed at source
        mask = [float(m) for m in attention_mask]
        
        # Identify opening and closing tag IDs
        opening_tag_ids = set()
        closing_tag_id = None
        
        for token_str, token_id in self.connector_token_ids.items():
            # Closing tag: </connector>
            if token_str == '</connector>':
                closing_tag_id = token_id
            # Opening tags: <connector type="X">
            elif token_str.startswith('<connector'):
                opening_tag_ids.add(token_id)
        
        # Mark connector spans with weight (only for non-padding positions)
        in_connector = False
        
        for i, token_id in enumerate(input_ids):
            # Skip if padding
            if attention_mask[i] == 0:
                mask[i] = 0.0
                continue
            
            # Entering connector region
            if token_id in opening_tag_ids:
                in_connector = True
                mask[i] = weight  # Mark opening tag with weight
            
            # Inside connector region
            elif in_connector:
                mask[i] = weight  # Mark all connector content
                
                # Exiting connector region
                if token_id == closing_tag_id:
                    in_connector = False
        
        return mask
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 2048,
        connector_weight: Optional[float] = None,
        pad_to_max: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize batch of texts with connector masks.
        
        Handles padding to create uniform batch shapes for GPU processing.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length per text
            connector_weight: Override weight for this batch
            pad_to_max: Whether to pad all sequences to max length in batch
        
        Returns:
            Dict with batched tensors:
            - 'input_ids': (batch_size, seq_len)
            - 'attention_mask': (batch_size, seq_len)
            - 'connector_mask': (batch_size, seq_len)
        """
        
        weight = connector_weight if connector_weight is not None else self.connector_weight
        
        batch_data = []
        
        # Process each text individually
        for text in texts:
            data = self.tokenize_with_connector_mask(
                text,
                max_length=max_length,
                return_tensors=None,  # Get as lists
                connector_weight=weight
            )
            batch_data.append(data)
        
        if not batch_data:
            return {'input_ids': torch.tensor([]), 'attention_mask': torch.tensor([]),
                   'connector_mask': torch.tensor([])}
        
        # Pad to same length
        if pad_to_max:
            max_len = max(len(d['input_ids']) for d in batch_data)
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_connector_mask = []
            
            for data in batch_data:
                pad_len = max_len - len(data['input_ids'])
                
                # Pad with 0 (which becomes padding token)
                padded_input_ids.append(data['input_ids'] + [0] * pad_len)
                padded_attention_mask.append(data['attention_mask'] + [0] * pad_len)
                # Pad connector mask with 0.0 (padding = no boost)
                padded_connector_mask.append(data['connector_mask'] + [0.0] * pad_len)
            
            return {
                'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
                'connector_mask': torch.tensor(padded_connector_mask, dtype=torch.float32)
            }
        else:
            # Return as-is (list of lists)
            return {
                'input_ids': batch_data,
                'attention_mask': batch_data,
                'connector_mask': batch_data
            }
    
    def set_connector_weight(self, weight: float):
        """
        Change the connector weight for this tokenizer instance.
        
        This allows experimenting with different weights during training.
        
        Args:
            weight: New weight value (e.g., 1.05, 1.1, 1.2, etc.)
        """
        
        self.connector_weight = weight
        logger.info(f"✓ Connector weight updated to {weight}x")
    
    def get_statistics(self) -> Dict:
        """
        Get information about this tokenizer instance.
        
        Returns:
            Dict with tokenizer statistics
        """
        
        return {
            'connector_weight': self.connector_weight,
            'special_tokens_count': len(self.connector_token_ids),
            'special_tokens': list(self.connector_token_ids.keys()),
            'vocab_size': len(self.tokenizer)
        }


# ============================================================================
# LEGACY FUNCTION FOR COMPATIBILITY
# ============================================================================

def tokenize_with_weighted_attention(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 2048,
    attention_weight: float = CONNECTOR_ATTENTION_WEIGHT,
    return_tensors: str = "pt"
) -> Dict:
    """
    Standalone function for tokenizing with connector attention weights.
    
    Legacy function for backwards compatibility with preprocess.py.
    
    Args:
        text: Input text with connector tags
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        attention_weight: Weight for connector tokens
        return_tensors: "pt" for PyTorch tensors
    
    Returns:
        Dict with 'input_ids', 'attention_mask', 'connector_mask'
    """
    
    connector_tok = ConnectorTokenizer(tokenizer, connector_weight=attention_weight)
    
    data = connector_tok.tokenize_with_connector_mask(
        text,
        max_length=max_length,
        return_tensors=return_tensors,
        connector_weight=attention_weight
    )
    
    return {
        'input_ids': data['input_ids'],
        'attention_mask': data['attention_mask'],
        'connector_mask': data['connector_mask'],
        'token_count': len(data['input_ids']) if isinstance(data['input_ids'], list) 
                      else data['input_ids'].shape[0]
    }


# ============================================================================
# TESTING & VERIFICATION
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import BASE_MODEL
    
    print("\n" + "="*80)
    print("TOKENIZER UTILS - CRD CONNECTOR MASK GENERATION")
    print("="*80)
    
    # Load and setup tokenizer
    logger.info(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    logger.info(f"✓ Added {num_added} special tokens")
    
    # Create connector tokenizer
    conn_tok = ConnectorTokenizer(tokenizer)
    
    # Test text with connector tag (exact format)
    test_text = 'The model works <connector type="CAUSAL"> because </connector> of good data.'
    
    logger.info(f"\n[Test 1] Single text with connector:")
    logger.info(f"  Input: {test_text}")
    
    # Tokenize with DEFAULT weight (1.1)
    logger.info(f"\n[Test 1] Tokenizing with DEFAULT weight: {CONNECTOR_ATTENTION_WEIGHT}")
    result = conn_tok.tokenize_with_connector_mask(test_text, return_tensors="pt")
    
    logger.info(f"  Sequence length: {len(result['input_ids'][0]) if result['input_ids'].dim() > 1 else len(result['input_ids'])}")
    mask_values = set(result['connector_mask'].tolist() if result['connector_mask'].dim() > 1 else result['connector_mask'].tolist())
    logger.info(f"  Unique mask values: {sorted(mask_values)}")
    
    # Verify
    assert 1.0 in mask_values, "ERROR: Default value 1.0 not found"
    if CONNECTOR_ATTENTION_WEIGHT > 1.0:
        assert CONNECTOR_ATTENTION_WEIGHT in mask_values, f"ERROR: {CONNECTOR_ATTENTION_WEIGHT} not found"
    logger.info(f"  ✓ Mask verified (connectors correctly weighted)")
    
    # Test 2: Batch with padding
    logger.info(f"\n[Test 2] Batch with padding:")
    texts = [
        "Short text with <connector type=\"CAUSAL\"> because </connector> reason.",
        "This is a much longer text to demonstrate padding. It goes on and on. And on. And has <connector type=\"TEMPORAL\"> when </connector> needed."
    ]
    
    batch = conn_tok.tokenize_batch(texts, pad_to_max=True)
    logger.info(f"  Batch shape: {batch['connector_mask'].shape}")
    
    # Check padding
    second_mask = batch['connector_mask'][0].tolist()
    padding_values = [m for m in second_mask if m == 0.0]
    logger.info(f"  First sequence padding zeros: {len(padding_values)} (expected: 0 or few)")
    
    second_mask = batch['connector_mask'][1].tolist()
    if len(second_mask) > len(texts[1].split()):
        padding_in_second = second_mask[-5:]
        logger.info(f"  Second sequence tail (includes padding): {padding_in_second}")
        assert all(v == 0.0 for v in padding_in_second), "ERROR: Padding not zeroed"
    logger.info(f"  ✓ Padding correctly handled (zeroed)")
    
    # Show statistics
    stats = conn_tok.get_statistics()
    logger.info(f"\n[Statistics]:")
    logger.info(f"  Connector weight: {stats['connector_weight']}")
    logger.info(f"  Special tokens: {stats['special_tokens_count']}")
    logger.info(f"  Vocab size: {stats['vocab_size']}")
    
    print("="*80 + "\n")