#!/usr/bin/env python3
"""
data_loader.py - ALIGNED WITH CONNECTOR BOOSTING DESIGN


Ultra memory-efficient loader with token ID validation.
Filters out samples with invalid token IDs to prevent CUDA errors.

CORRECTED: Proper tag format detection and connector_mask creation
that aligns with our 1.1x hidden state boosting design.
"""


import logging
import glob
import os
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from typing import Optional, List, Dict
from transformers import PreTrainedTokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# ============================================================================
# Connector Data Collator - CORRECTED
# ============================================================================


class ConnectorDataCollatorWithMaskCreation:
    """
    Collate batch and CREATE connector_mask on-the-fly.
    
    CORRECTED: Proper detection of <connector type="x">word</connector> format.
    Aligns with our 1.1x hidden state boosting design.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        pad_token_id: int,
        boost_factor: float = 1.1
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.boost_factor = boost_factor
        self.vocab_size = len(tokenizer)
        
        self.opening_tag_ids = set()
        self.closing_tag_id = None
        
        logger.info("Initializing connector tag token IDs:")
        
        # Get connector types from config
        try:
            from utils.config import Config
            config = Config()
            connector_types = list(config.connector_types.keys())
        except:
            connector_types = ['causal', 'conclusive', 'contrastive', 'temporal', 'additive', 'adversative']
        
        # CORRECTED: Use full tag format <connector type="x">
        for conn_type in connector_types:
            # Full tag format as per our design
            tag = f'<connector type="{conn_type}">'
            try:
                token_id = tokenizer.convert_tokens_to_ids(tag)
                if token_id != tokenizer.unk_token_id:
                    self.opening_tag_ids.add(token_id)
                    logger.info(f"  Opening tag: {tag:<40} → ID {token_id}")
                else:
                    logger.warning(f"  ⚠ Opening tag not in vocab: {tag}")
            except Exception as e:
                logger.warning(f"  ⚠ Error with tag {tag}: {e}")
        
        # Closing tag
        try:
            closing_tag = "</connector>"
            self.closing_tag_id = tokenizer.convert_tokens_to_ids(closing_tag)
            if self.closing_tag_id == tokenizer.unk_token_id:
                self.closing_tag_id = None
                logger.warning(f"  ⚠ Closing tag not in vocab: {closing_tag}")
            else:
                logger.info(f"  Closing tag: {closing_tag:<40} → ID {self.closing_tag_id}")
        except Exception as e:
            logger.warning(f"  ⚠ Error getting closing tag: {e}")
        
        if not self.opening_tag_ids or not self.closing_tag_id:
            logger.warning("\n⚠️  WARNING: Connector tags not properly initialized!")
            logger.warning("   This may indicate tokenizer mismatch or incomplete setup.")
    
    def _create_connector_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create connector_mask for hidden state boosting.
        
        Returns:
            connector_mask: [batch_size, seq_len]
                          Padding: 1.0 (no boost)
                          Real tokens: 1.0 (normal) or boost_factor (connector)
        """
        batch_size, seq_len = input_ids.shape
        connector_mask = torch.ones_like(input_ids, dtype=torch.float)
        
        for b in range(batch_size):
            inside_connector = False
            
            for pos in range(seq_len):
                token_id = input_ids[b, pos].item()
                is_real_token = attention_mask[b, pos].item() == 1
                
                # Padding tokens: keep mask as 1.0 (no boost)
                if not is_real_token:
                    connector_mask[b, pos] = 1.0
                    continue
                
                # Real tokens: detect connector tags
                if token_id in self.opening_tag_ids:
                    inside_connector = True
                    connector_mask[b, pos] = self.boost_factor  # Boost opening tag
                
                elif self.closing_tag_id and token_id == self.closing_tag_id:
                    connector_mask[b, pos] = self.boost_factor  # Boost closing tag
                    inside_connector = False
                
                elif inside_connector:
                    connector_mask[b, pos] = self.boost_factor  # Boost words inside connector
        
        return connector_mask
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch and create masks.
        
        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Binary mask (0/1) for transformer
            - connector_mask: Boost mask (1.0/boost_factor) for hidden states
            - labels: For causal LM loss
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        # Create labels for causal LM (next-token prediction)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        
        # Create connector_mask for hidden state boosting
        connector_mask = self._create_connector_mask(input_ids, attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,  # Binary: 0/1 for transformer
            "connector_mask": connector_mask,  # Boost: 1.0/boost_factor for hidden states
            "labels": labels
        }



# ============================================================================
# Data Loader with Token Validation
# ============================================================================


class PretrainingDataLoader:
    """
    Ultra memory-efficient loader with token ID validation.
    
    FEATURES:
    - Memory-efficient loading (saves to disk incrementally)
    - Filters invalid token IDs (prevents CUDA errors)
    - Caching for fast subsequent loads
    - Compatible with connector boosting pipeline
    """
    
    def __init__(self, config, data_dir: str = "data_splits", cache_dir: str = ".cache/datasets"):
        self.config = config
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.vocab_size = None
        logger.info(f"Initialized PretrainingDataLoader")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Cache dir: {self.cache_dir}")
    
    def _filter_invalid_tokens(self, dataset: Dataset, vocab_size: int) -> Dataset:
        """
        Filter out samples with token IDs >= vocab_size.
        
        Prevents CUDA device-side assert errors during training.
        """
        logger.info(f"\n🔍 Validating token IDs (vocab_size={vocab_size})...")
        
        def is_valid_sample(example):
            """Check if all token IDs are < vocab_size"""
            input_ids = example['input_ids']
            if not input_ids:
                return False
            max_id = max(input_ids)
            return max_id < vocab_size
        
        total = len(dataset)
        logger.info(f"  Checking {total:,} samples...")
        
        # Filter dataset
        filtered_dataset = dataset.filter(is_valid_sample, num_proc=4)
        
        invalid_count = total - len(filtered_dataset)
        
        if invalid_count > 0:
            logger.warning(f"\n⚠️  FILTERED OUT {invalid_count:,} samples with invalid token IDs")
            logger.warning(f"   These samples had token IDs >= {vocab_size}")
            logger.warning(f"   Kept {len(filtered_dataset):,} valid samples ({len(filtered_dataset)/total*100:.1f}%)")
            logger.warning(f"\n💡 To fix: Re-preprocess with correct tokenizer")
        else:
            logger.info(f"  ✓ All {total:,} samples have valid token IDs")
        
        return filtered_dataset
    
    def load_preprocessed_data_for_training(
        self,
        test_split_size: float = 0.05,
        seed: int = 42,
        max_chunks: Optional[int] = None,
        use_cache: bool = True,
        vocab_size: Optional[int] = None
    ) -> Optional[DatasetDict]:
        """
        Load training data with token validation.
        
        Args:
            test_split_size: Proportion for test split
            seed: Random seed for shuffle
            max_chunks: Limit number of chunks to load
            use_cache: Use cached dataset if available
            vocab_size: Vocabulary size (auto-detected if None)
        
        Returns:
            DatasetDict with validated 'train' and 'test' datasets
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING TRAINING DATA (WITH VALIDATION)")
        logger.info("="*80)
        
        # Auto-detect vocab size
        if vocab_size is None:
            try:
                from utils.config import Config
                from transformers import AutoTokenizer
                
                cfg = Config()
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
                special_tokens = cfg.get_special_tokens()
                tokenizer.add_tokens(special_tokens)
                vocab_size = len(tokenizer)
                logger.info(f"✓ Auto-detected vocab size: {vocab_size:,}")
            except Exception as e:
                logger.error(f"⚠️  Could not auto-detect vocab size: {e}")
                return None
        
        self.vocab_size = vocab_size
        
        # Check cache
        cache_path = Path(self.cache_dir) / "combined_dataset"
        if use_cache and cache_path.exists():
            logger.info(f"\n✓ Found cached dataset at {cache_path}")
            logger.info("Loading from cache...")
            try:
                full_dataset = load_from_disk(str(cache_path))
                logger.info(f"✓ Loaded {len(full_dataset):,} samples from cache")
                
                # Validate token IDs (security check)
                full_dataset = self._filter_invalid_tokens(full_dataset, vocab_size)
                
                # Create split
                logger.info(f"\nCreating train/test split ({test_split_size*100:.1f}% test)...")
                dataset_dict = full_dataset.train_test_split(
                    test_size=test_split_size,
                    seed=seed,
                    shuffle=True
                )
                
                logger.info(f"  Train: {len(dataset_dict['train']):,}")
                logger.info(f"  Test: {len(dataset_dict['test']):,}")
                logger.info("\n" + "="*80)
                logger.info("✓ Data loading complete (from cache)")
                logger.info("="*80)
                
                return dataset_dict
            except Exception as e:
                logger.warning(f"⚠️  Could not load cache: {e}")
                logger.info("Will rebuild dataset...")
        
        try:
            # Find files
            train_pattern = str(Path(self.data_dir) / "train_chunk_*.parquet")
            train_files = sorted(glob.glob(train_pattern))
            
            if not train_files:
                logger.error(f"✗ No train_chunk files found: {train_pattern}")
                return None
            
            logger.info(f"\nFound {len(train_files)} train chunk files")
            
            # Limit chunks if requested
            if max_chunks is not None:
                train_files = train_files[:max_chunks]
                logger.info(f"⚠️  Loading only first {max_chunks} chunks")
            
            # Process chunks incrementally
            logger.info(f"\nProcessing {len(train_files)} chunks...")
            logger.info("Saving incrementally to avoid memory overflow...")
            
            temp_dir = Path(self.cache_dir) / "temp_chunks"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            total_samples = 0
            saved_chunks = []
            
            for i, train_file in enumerate(train_files, 1):
                try:
                    logger.info(f"\n[{i}/{len(train_files)}] Processing {Path(train_file).name}...")
                    
                    df = pd.read_parquet(train_file)
                    chunk_samples = len(df)
                    logger.info(f"  Loaded: {chunk_samples:,} samples")
                    
                    # Convert to Dataset
                    chunk_dataset = Dataset.from_pandas(df, preserve_index=False)
                    del df
                    
                    # Save to disk immediately (memory efficient)
                    chunk_save_path = temp_dir / f"chunk_{i:04d}"
                    chunk_dataset.save_to_disk(str(chunk_save_path))
                    saved_chunks.append(str(chunk_save_path))
                    
                    total_samples += chunk_samples
                    logger.info(f"  ✓ Saved to disk (total: {total_samples:,} samples)")
                    
                    del chunk_dataset
                    
                except Exception as e:
                    logger.warning(f"  ✗ Error: {e}")
                    continue
            
            if not saved_chunks:
                logger.error("✗ Could not process any chunks")
                return None
            
            # Load and concatenate from disk
            logger.info(f"\n✓ Processed {len(saved_chunks)} chunks ({total_samples:,} samples)")
            logger.info("Loading saved chunks and concatenating...")
            
            datasets = []
            for i, chunk_path in enumerate(saved_chunks, 1):
                logger.info(f"  [{i}/{len(saved_chunks)}] Loading {Path(chunk_path).name}...")
                chunk = load_from_disk(chunk_path)
                datasets.append(chunk)
            
            logger.info("Concatenating all chunks...")
            full_dataset = concatenate_datasets(datasets)
            del datasets
            
            logger.info(f"✓ Combined dataset: {len(full_dataset):,} samples")
            
            # CRITICAL: Validate token IDs before caching
            logger.info("\n" + "="*80)
            full_dataset = self._filter_invalid_tokens(full_dataset, vocab_size)
            logger.info("="*80)
            
            # Save to cache
            logger.info(f"\nSaving validated dataset to cache: {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            full_dataset.save_to_disk(str(cache_path))
            logger.info("✓ Cached for future use")
            
            # Cleanup temp files
            logger.info("\nCleaning up temporary files...")
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info("✓ Cleanup complete")
            except Exception as e:
                logger.warning(f"⚠️  Could not cleanup temp dir: {e}")
            
            # Verify columns
            logger.info(f"\nDataset columns: {full_dataset.column_names}")
            required_cols = ['input_ids', 'attention_mask']
            missing_cols = [col for col in required_cols if col not in full_dataset.column_names]
            
            if missing_cols:
                logger.error(f"✗ Missing required columns: {missing_cols}")
                return None
            
            # Create train/test split
            logger.info(f"\nCreating train/test split ({test_split_size*100:.1f}% test)...")
            dataset_dict = full_dataset.train_test_split(
                test_size=test_split_size,
                seed=seed,
                shuffle=True
            )
            
            logger.info(f"  Train: {len(dataset_dict['train']):,}")
            logger.info(f"  Test: {len(dataset_dict['test']):,}")
            
            logger.info("\n" + "="*80)
            logger.info("✓ Data loading complete")
            logger.info("="*80)
            logger.info("\n💡 Next time will be MUCH faster (loading from cache)")
            logger.info(f"💡 Connector boosting enabled: boost_factor=1.1")
            
            return dataset_dict
        
        except Exception as e:
            logger.error(f"\n✗ Error: {e}", exc_info=True)
            return None
    
    def clear_cache(self):
        """Clear cached dataset and temporary files."""
        cache_path = Path(self.cache_dir) / "combined_dataset"
        temp_path = Path(self.cache_dir) / "temp_chunks"
        
        import shutil
        
        for path in [cache_path, temp_path]:
            if path.exists():
                try:
                    shutil.rmtree(path)
                    logger.info(f"✓ Cleared: {path}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not clear {path}: {e}")
        
        logger.info("Cache cleanup complete")



if __name__ == "__main__":
    try:
        from utils.config import Config
        
        config = Config()
        data_loader = PretrainingDataLoader(config)
        
        logger.info("\n[Test] Loading with token validation and connector mask creation...")
        dataset_dict = data_loader.load_preprocessed_data_for_training(max_chunks=2)
        
        if dataset_dict:
            logger.info("\n✓ Success!")
            logger.info(f"  Train: {len(dataset_dict['train']):,}")
            logger.info(f"  Test: {len(dataset_dict['test']):,}")
            logger.info(f"\n✓ Connector boosting masks will be created on-the-fly during training")
    
    except ImportError:
        logger.error("Run from project root")
    except Exception as e:
        logger.error(f"Error: {e}")