#!/usr/bin/env python3
"""
data_loader_fixed_final.py - COLLATOR THAT HANDLES LISTS → TENSORS

FIXED: The collator now converts lists to tensors automatically.

The issue:
- Collator expects: tensor inputs
- Collator receives: list inputs (because HF Trainer bypasses wrappers)
- Result: torch.stack fails on lists

Solution:
- Check if input is list or tensor.
- Convert lists to tensors IN THE COLLATOR
- Then proceed with stacking
"""

import logging
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from typing import Optional, List, Dict
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConnectorDataCollatorWithMaskCreation:
    """
    FIXED: Collator that handles list inputs and converts to tensors.
    
    Key fix:
    - Checks if batch items are lists or tensors
    - Converts lists to tensors automatically
    - Handles string values (converts to int)
    - Creates connector masks
    """
    
    def __init__(self, tokenizer, pad_token_id=None, boost_factor=1.1):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id
        self.boost_factor = boost_factor
        
        # Initialize connector tag detection
        self._init_connector_tags()
    
    def _init_connector_tags(self):
        """Initialize connector tag token IDs from tokenizer."""
        logger.info("Initializing connector tag token IDs:")
        
        connector_opening_tags = set()
        connector_closing_tag_id = None
        
        # Expected tag formats (from config.py)
        opening_tag_templates = [
            '<connector type="CAUSAL">',
            '<connector type="ADVERSATIVE">',
            '<connector type="TEMPORAL">',
            '<connector type="CONDITIONAL">',
            '<connector type="CONCLUSIVE">',
            '<connector type="ADDITIVE">',
        ]
        
        closing_tag_str = '</connector>'
        
        # Extract token IDs from tokenizer
        for tag_str in opening_tag_templates:
            try:
                token_ids = self.tokenizer.encode(tag_str, add_special_tokens=False)
                if len(token_ids) == 1:
                    tag_id = token_ids[0]
                    connector_opening_tags.add(tag_id)
                    logger.info(f"  Opening tag: {tag_str:40} → ID {tag_id}")
                else:
                    logger.warning(f"  ⚠ Opening tag encoded to multiple tokens: {tag_str}")
            except Exception as e:
                logger.warning(f"  ⚠ Error encoding tag {tag_str}: {e}")
        
        # Extract closing tag
        try:
            token_ids = self.tokenizer.encode(closing_tag_str, add_special_tokens=False)
            if len(token_ids) == 1:
                connector_closing_tag_id = token_ids[0]
                logger.info(f"  Closing tag: {closing_tag_str:40} → ID {connector_closing_tag_id}")
            else:
                logger.warning(f"  ⚠ Closing tag encoded to multiple tokens: {token_ids}")
        except Exception as e:
            logger.warning(f"  ⚠ Error encoding closing tag: {e}")
        
        self.connector_opening_tags = connector_opening_tags
        self.connector_closing_tag_id = connector_closing_tag_id
        
        if connector_opening_tags and connector_closing_tag_id:
            logger.info(f"✓ Found {len(connector_opening_tags)} opening tags + 1 closing tag")
        else:
            logger.warning(f"\n⚠️  WARNING: Connector tags not properly initialized!")
            logger.warning(f"   This may indicate tokenizer mismatch or incomplete setup.")
    
    def _ensure_int_list(self, values):
        """
        FIXED: Convert values to integer list.
        
        Handles:
        - None → []
        - Strings → ints
        - Already ints → pass through
        """
        if values is None:
            return []
        
        if isinstance(values, list):
            if len(values) == 0:
                return []
            
            # If strings, convert to ints
            if isinstance(values[0], str):
                try:
                    return [int(v) for v in values]
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert list values to int: {values[:3]}")
                    return values
            return values
        
        return []
    
    def _create_boost_mask(self, input_ids, connector_types_mapping=None):
        """
        ✅ SMART: Create connector boost mask by detecting connector tags in input_ids.
        
        Algorithm:
        1. Scan input_ids for connector opening tags (128010-128015)
        2. When found, identify the connector type from token ID
        3. Mark all tokens between opening and closing tag with 1.1
        4. All other tokens marked 1.0
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        
        if not input_ids or len(input_ids) == 0:
            return torch.tensor([[1.0]], dtype=torch.float32)
        
        # Determine batch/seq dimensions
        if isinstance(input_ids, list):
            batch_size = len(input_ids)
            seq_len = len(input_ids)
        else:
            batch_size = 1
            seq_len = len(input_ids)
        
        # Initialize mask: all 1.0 (no boosting)
        mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        
        # ✅ CONNECTOR TOKEN IDS (mapped from special tokens added to vocabulary)
        # These correspond to the opening tags added to tokenizer
        CONNECTOR_TAG_IDS = {
            128257: 'CAUSAL',      # <connector type="CAUSAL">
            128258: 'ADVERSATIVE', # <connector type="ADVERSATIVE">
            128259: 'TEMPORAL',    # <connector type="TEMPORAL">
            128260: 'CONDITIONAL', # <connector type="CONDITIONAL">
            128261: 'CONCLUSIVE',  # <connector type="CONCLUSIVE">
            128262: 'ADDITIVE',    # <connector type="ADDITIVE">
        }
        
        CLOSING_TAG_ID = 128263  # </connector>
        
        # Process single or batch
        if isinstance(input_ids, list):
            # Batch mode
            for batch_idx, seq in enumerate(input_ids):
                i = 0
                while i < len(seq):
                    token_id = seq[i]
                    
                    # Check if this is a connector opening tag
                    if token_id in CONNECTOR_TAG_IDS:
                        connector_type = CONNECTOR_TAG_IDS[token_id]
                        
                        # Mark opening tag
                        mask[batch_idx, i] = 1.1
                        
                        # Mark all tokens until closing tag
                        i += 1
                        while i < len(seq):
                            mask[batch_idx, i] = 1.1
                            if seq[i] == CLOSING_TAG_ID:
                                break
                            i += 1
                    
                    i += 1
        else:
            # Single sequence mode
            i = 0
            while i < len(input_ids):
                token_id = input_ids[i]
                
                # Check if this is a connector opening tag
                if token_id in CONNECTOR_TAG_IDS:
                    connector_type = CONNECTOR_TAG_IDS[token_id]
                    
                    # Mark opening tag
                    mask[0, i] = 1.1
                    
                    # Mark all tokens until closing tag
                    i += 1
                    while i < len(input_ids):
                        mask[0, i] = 1.1
                        if input_ids[i] == CLOSING_TAG_ID:
                            break
                        i += 1
                
                i += 1
        
        return mask
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """FINAL: Collate batch - ONLY return what's needed!"""
        
        if not batch:
            raise ValueError("Empty batch")
        
        # Extract ONLY input_ids and attention_mask
        input_ids_list = []
        attention_mask_list = []
        
        for item in batch:
            input_ids = item.get("input_ids")
            attention_mask = item.get("attention_mask")
            
            # Handle tensors/lists
            if isinstance(input_ids, torch.Tensor):
                input_ids = input_ids.tolist()
            input_ids = self._ensure_int_list(input_ids)
            
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.tolist()
            attention_mask = self._ensure_int_list(attention_mask)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 1
        
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            if pad_len > 0:
                input_ids_list[i].extend([self.pad_token_id] * pad_len)
                attention_mask_list[i].extend([0] * pad_len)
        
        # Convert to tensors
        input_ids_tensor = torch.stack([
            torch.tensor(ids, dtype=torch.long) for ids in input_ids_list
        ])
        attention_mask_tensor = torch.stack([
            torch.tensor(mask, dtype=torch.long) for mask in attention_mask_list
        ])
        
        # Create labels
        labels = input_ids_tensor.clone()
        labels[attention_mask_tensor == 0] = -100
        
        # ✅ RETURN ONLY 3 KEYS - NO connector_mask!
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
        }


# ============================================================================
# Direct Parquet Dataset (no changes needed)
# ============================================================================

class DirectParquetDataset:
    """Load parquet files with string-to-integer conversion."""
    
    def __init__(self, parquet_path: str, max_files: Optional[int] = None):
        self.parquet_path = Path(parquet_path)
        
        # Find parquet files
        if self.parquet_path.is_file():
            parquet_files = [self.parquet_path]
        else:
            parquet_files = sorted(self.parquet_path.glob("*.parquet"))
        
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        self.data = []
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)
            
            # Validate columns
            required_cols = {'input_ids', 'attention_mask'}
            actual_cols = set(df.columns)
            
            if not required_cols.issubset(actual_cols):
                logger.warning(f"File {file_path.name} missing columns: {required_cols - actual_cols}")
                continue
            
            # Convert to list of dicts
            self.data.extend(df.to_dict('records'))
        
        if not self.data:
            raise ValueError("No valid data loaded from parquet files")
        
        logger.info(f"Loaded {len(self.data):,} samples from parquet")
        self.total_samples = len(self.data)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Get item by index."""
        item = self.data[idx]
        
        # Extract fields
        input_ids = item.get("input_ids", [])
        attention_mask = item.get("attention_mask", [])
        # connector_mask = item.get("connector_mask")
        
        # Ensure lists
        if isinstance(input_ids, str):
            try:
                import ast
                input_ids = ast.literal_eval(input_ids)
            except:
                input_ids = []
        
        if isinstance(attention_mask, str):
            try:
                import ast
                attention_mask = ast.literal_eval(attention_mask)
            except:
                attention_mask = []
        
        # if isinstance(connector_mask, str):
        #     try:
        #         import ast
        #         connector_mask = ast.literal_eval(connector_mask)
        #     except:
        #         connector_mask = None
        
        # Ensure proper types
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], str):
            input_ids = [int(x) for x in input_ids]
        
        if isinstance(attention_mask, list) and len(attention_mask) > 0 and isinstance(attention_mask[0], str):
            attention_mask = [int(x) for x in attention_mask]
        
        # if connector_mask is not None and isinstance(connector_mask, list) and len(connector_mask) > 0 and isinstance(connector_mask[0], str):
        #     connector_mask = [float(x) for x in connector_mask]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "connector_mask": connector_mask,
            "connector_words": item.get("connector_words", []),
            "connector_types": item.get("connector_types", []),
        }


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("data_loader_fixed_final.py - COLLATOR THAT HANDLES LIST → TENSOR")
    logger.info("=" * 80)
    logger.info("Key fix: Collator now converts lists to tensors automatically")
    logger.info("=" * 80)
