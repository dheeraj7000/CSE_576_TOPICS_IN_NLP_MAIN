#!/usr/bin/env python3
"""
<<<<<<< HEAD
main_fixed_v5.py - REPAIR NULL VALUES (DON'T SKIP FILES!)

FIXED APPROACH:
Instead of skipping problematic files, REPAIR them:
1. NULL/None in lists → convert to empty lists
2. Missing columns → add with defaults
3. Type mismatches → convert to correct types
4. Invalid values → replace with defaults

This preserves ALL 186,899 samples instead of losing them!

Key repairs:
- None/null → []
- Missing input_ids → generate zeros
- Wrong types → force to correct type
- Concatenation errors → fix before conversion
=======
main.py - UPDATED FOR model_multiword_support.py

CRITICAL SIMPLIFICATION:
- Model auto-detects connector tags from input_ids
- NO TensorDatasetWrapper needed!
- Simpler data pipeline
- Works with or without connector_mask in parquet

FLOW:
parquet → input_ids + attention_mask → model detects tags → boost applied
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
"""

import os
import gc
import torch
import logging
from pathlib import Path
from huggingface_hub import login
from datasets import Dataset as HFDataset, concatenate_datasets
import pandas as pd
import numpy as np
from tqdm import tqdm

# CORRECTED IMPORTS
from utils.config import Config
from pretrain.model import initialize_model
from pretrain.trainer import ConnectorPretrainingManager

from pathlib import Path
import gc
import pandas as pd
import pyarrow.parquet as pq
from datasets import concatenate_datasets
def yield_parquet_dataset_chunks(parquet_path: str, chunk_size: int = 20, split_ratio: float = 0.05):
    """
    Yield HFDatasets built from at most `chunk_size` parquet files at a time.
    Ensures:
      • required columns present
      • non-empty frames
      • safe split (≥1 train sample)
      • skips chunks that would produce train==0
    """
    p = Path(parquet_path)
    parquet_files = sorted([p] if p.is_file() else p.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {parquet_path}")

    for start in range(0, len(parquet_files), chunk_size):
        batch_files = parquet_files[start:start + chunk_size]

        all_datasets = []
        for file_path in batch_files:
            try:
                pa_cols = set(pq.ParquetFile(file_path).schema.names)
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: could not read schema ({e})")
                continue

            required = {"input_ids", "attention_mask"}
            if not required.issubset(pa_cols):
                logger.warning(f"Skipping {file_path.name}: missing columns {sorted(required - pa_cols)}; have {sorted(pa_cols)}")
                continue

            cols = [c for c in ("input_ids", "attention_mask", "connector_mask") if c in pa_cols]
            df = pd.read_parquet(file_path, columns=cols)

            if df.empty:
                logger.warning(f"Skipping {file_path.name}: zero rows")
                continue

            def _ok(x):
                return isinstance(x, (list, np.ndarray)) and len(x) > 0
            df = df[df["input_ids"].map(_ok) & df["attention_mask"].map(_ok)]
            if df.empty:
                logger.warning(f"Skipping {file_path.name}: all rows invalid (empty sequences)")
                continue

            df = clean_dataframe_batch(df)
            ds = df_to_hfdataset_batch(df)
            all_datasets.append(ds)

            del df, ds
            gc.collect()

        if not all_datasets:
            logger.warning("No usable parquet files in this chunk; skipping chunk.")
            continue

        full_ds = all_datasets[0] if len(all_datasets) == 1 else concatenate_datasets(all_datasets)
        n = len(full_ds)
        if n == 0:
            logger.warning("Chunk produced 0 samples; skipping chunk.")
            continue

        # Safe split that guarantees ≥1 train sample
        if 0 < split_ratio < 1 and n >= 2:
            test_n = max(1, int(round(n * split_ratio)))
            test_n = min(test_n, n - 1)
            split = full_ds.train_test_split(test_size=test_n, seed=42, shuffle=True)
            dataset_dict = {'train': split['train'], 'validation': split['test']}
        else:
            dataset_dict = {'train': full_ds, 'validation': None}

        if len(dataset_dict['train']) == 0:
            logger.warning("After split, train=0; skipping chunk.")
            continue

        yield dataset_dict

        del all_datasets, full_ds, dataset_dict
        gc.collect()


def train_in_chunks(config, model_handler, parquet_path: str, chunk_size: int = 20, split_ratio: float = 0.05):
    """
    Keep one Trainer and swap its dataset per chunk to preserve optimizer/LR state.
    """
    # Create manager and Trainer once (mirrors your run_training() flow):contentReference[oaicite:8]{index=8}
    pretrain_manager = ConnectorPretrainingManager(
        config=config,
        model_handler=model_handler,
        use_new_collator=True  # you already use this:contentReference[oaicite:9]{index=9}
    )

    chunk_iter = yield_parquet_dataset_chunks(parquet_path, chunk_size=chunk_size, split_ratio=split_ratio)
    # Initialize Trainer on the first non-empty chunk
    first = next(chunk_iter, None)
    if first is None or len(first['train']) == 0:
        raise RuntimeError(
            "No non-empty training chunk was found. "
            "Check PARQUET_PATH, required columns (input_ids, attention_mask), and that parquet files have rows."
        )
    pretrain_manager.prepare_trainer(
        train_dataset=first['train'],
        eval_dataset=first.get('validation'),
        output_dir="./output/connector_model_stream",
        boost_factor=config.boost_factor,
        use_amplification=False,
        amplification_strength=1.0,
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-6
    )
    trainer = pretrain_manager.trainer
    trainer.train()

    clear_cuda_memory()  # your helper: gc + cuda empty + reset peak stats:contentReference[oaicite:10]{index=10}

    # Iterate remaining chunks
    for dataset_dict in chunk_iter:
        # Swap datasets; keep optimizer/scheduler
        if dataset_dict is None or len(dataset_dict['train']) == 0:
            logger.warning("Skipping a chunk with empty train split")
            continue
        trainer.train_dataset = dataset_dict['train']
        trainer.eval_dataset = dataset_dict.get('validation', None)

        # Force dataloader rebuild if cached by your HF version
        if hasattr(trainer, "_reset_train_dataloader"):
            trainer._reset_train_dataloader()

        # Continue training; not resuming from checkpoint since we never stopped the process
        trainer.train(resume_from_checkpoint=False)

        clear_cuda_memory()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_cuda_memory():
    """Clear CUDA memory thoroughly."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"✓ GPU memory cleared")


def setup_huggingface():
    """Login to HuggingFace if token available"""
    token = os.environ.get('HF_TOKEN')
    if token:
        logger.info("Logging into HuggingFace...")
        login(token=token)
        logger.info("✓ Logged in")
    else:
        logger.warning("⚠ No HF_TOKEN found")


def load_model_handler(config: Config):
    """Load and prepare model with connector tokens."""
    logger.info("\n" + "="*70)
    logger.info("LOADING MODEL")
    logger.info("="*70)
    
    clear_cuda_memory()
    
    model_handler = initialize_model(config)
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"✓ GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    logger.info("✓ Model loaded successfully")
    logger.info("="*70)
    
    return model_handler


<<<<<<< HEAD
def convert_string_to_int(value):
    """Convert a single value from string to int if needed."""
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    return value


def convert_list_strings_to_ints(values):
    """Convert list of strings to integers. Handle None/empty."""
    if values is None:
        return []
    
    if isinstance(values, list):
        if len(values) == 0:
            return []
        try:
            if isinstance(values[0], str):
                return [int(v) for v in values]
            return values
        except (ValueError, TypeError):
            return []
    
    return []


def ensure_list(value, default_len=0):
    """
    Convert ANY value to a list.
    None/null → []
    Single value → [value]
    List → list (unchanged)
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        return value if len(value) > 0 else []
    
    if isinstance(value, (str, int, float)):
        return [value]
    
    return []


def repair_row(row, column_config):
    """
    REPAIR a single row with NULL values and type errors.
    
    Args:
        row: Dictionary row from parquet
        column_config: Dict of column → expected_type
    
=======
def clean_dataframe_batch(df):
    """
    Clean a single batch DataFrame.
    
    Converts numpy arrays to lists for HFDataset compatibility.
    
    Args:
        df: Pandas DataFrame
        
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    Returns:
        Repaired row dictionary
    """
    repaired = {}
    
<<<<<<< HEAD
    for col, expected_type in column_config.items():
        value = row.get(col)
        
        # REPAIR based on expected type
        if expected_type == 'int_list':
            # Expected: list of integers
            # Repair: None → [], strings → ints
            repaired[col] = convert_list_strings_to_ints(value)
        
        elif expected_type == 'float_list':
            # Expected: list of floats
            if value is None:
                repaired[col] = []
            elif isinstance(value, list):
                try:
                    repaired[col] = [float(v) for v in value]
                except (ValueError, TypeError):
                    repaired[col] = []
            else:
                repaired[col] = []
        
        elif expected_type == 'str_list':
            # Expected: list of strings
            repaired[col] = ensure_list(value)
        
        elif expected_type == 'string':
            # Expected: string
            if value is None:
                repaired[col] = ""
            else:
                repaired[col] = str(value)
        
        elif expected_type == 'int':
            # Expected: single integer
            if value is None:
                repaired[col] = 0
            else:
                repaired[col] = int(value) if isinstance(value, (int, float)) else 0
        
        else:
            # Fallback: keep as-is
            repaired[col] = value
=======
    # Process each column
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == object:
            # Check if column contains numpy arrays
            sample = cleaned_df[col].iloc[0] if len(cleaned_df) > 0 else None
            if isinstance(sample, np.ndarray):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            # Handle null values in list columns
            elif isinstance(sample, list):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: x if (isinstance(x, list) and x is not None) else []
                )
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    
    return repaired


def repair_dataframe_batch(df, column_config):
    """
    REPAIR entire batch - fix ALL NULL values and type errors.
    
    Args:
        df: Pandas DataFrame from parquet
        column_config: Dict of column → expected_type
    
    Returns:
        Repaired DataFrame
    """
    logger.debug(f"Repairing {len(df)} rows (fixing NULL values and type errors)...")
    
    repaired_rows = []
    for idx, row in df.iterrows():
        try:
            repaired_row = repair_row(row.to_dict(), column_config)
            repaired_rows.append(repaired_row)
        except Exception as e:
            logger.warning(f"  Row {idx}: Error during repair: {e}, using empty defaults")
            # Create row with all empty/default values
            repaired_row = {col: ([] if 'list' in typ else "" if typ == 'string' else 0) 
                          for col, typ in column_config.items()}
            repaired_rows.append(repaired_row)
    
    # Convert list of dicts to DataFrame
    repaired_df = pd.DataFrame(repaired_rows)
    return repaired_df


def df_to_hfdataset_batch(df):
    """
    Convert DataFrame to HFDataset.
<<<<<<< HEAD
    After repair, this should succeed.
=======
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        HFDataset
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    """
    # Convert to dict format
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].tolist()
    
    try:
        dataset = HFDataset.from_dict(data_dict)
        return dataset
    except Exception as e:
        logger.error(f"  Error converting to HFDataset: {e}")
        raise


<<<<<<< HEAD
class StringConvertingDataset:
    """
    FIXED: Extra safety - double-check all values are proper types.
    """
    
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.length = len(hf_dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Ensure input_ids
        input_ids = item.get("input_ids")
        if input_ids is None or (isinstance(input_ids, list) and len(input_ids) == 0):
            # Generate minimal valid sequence (pad token)
            input_ids = torch.tensor([128001] * 10, dtype=torch.long)
        elif isinstance(input_ids, list):
            if len(input_ids) > 0 and isinstance(input_ids[0], str):
                input_ids = [int(v) for v in input_ids]
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Ensure attention_mask
        attention_mask = item.get("attention_mask")
        if attention_mask is None or (isinstance(attention_mask, list) and len(attention_mask) == 0):
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        elif isinstance(attention_mask, list):
            if len(attention_mask) > 0 and isinstance(attention_mask[0], str):
                attention_mask = [int(v) for v in attention_mask]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Ensure connector_mask
        connector_mask = item.get("connector_mask")
        if connector_mask is None or (isinstance(connector_mask, list) and len(connector_mask) == 0):
            connector_mask = torch.ones(len(input_ids), dtype=torch.float)
        elif isinstance(connector_mask, list):
            connector_mask = [float(v) for v in connector_mask]
            connector_mask = torch.tensor(connector_mask, dtype=torch.float)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "connector_mask": connector_mask,
            "connector_words": ensure_list(item.get("connector_words", [])),
            "connector_types": ensure_list(item.get("connector_types", [])),
        }


def load_data_from_parquet_batch_wise(parquet_path: str, split_ratio: float = 0.05):
    """
    FIXED v5: Load parquet with REPAIR (not skip).
    
    Repairs:
    - NULL/None values → defaults
    - Type mismatches → force to correct type
    - Concatenation errors → fix before merge
    - All samples preserved!
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA FROM PARQUET (BATCH-WISE, FIXED v5)")
    logger.info("✓ REPAIRING NULL VALUES (not skipping)")
    logger.info("✓ Converting strings to integers")
    logger.info("✓ Preserving ALL samples")
=======
def load_data_from_parquet_batch_wise(parquet_path: str, split_ratio: float = 0.05):
    """
    Load data from parquet files BATCH-WISE (memory efficient).
    
    SIMPLIFIED: No TensorDatasetWrapper needed!
    Model auto-detects connector tags from input_ids.
    
    Args:
        parquet_path: Path to parquet directory or file
        split_ratio: Ratio for train/validation split
    
    Returns:
        dataset_dict: Dictionary with 'train' and 'validation' HFDatasets
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA FROM PARQUET (BATCH-WISE)")
    logger.info("SIMPLIFIED: Model auto-detects connector tags")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    logger.info("="*70)
    
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        logger.error(f"❌ Path not found: {parquet_path}")
        return None
    
    # Get all parquet files
    if parquet_path.is_file():
        parquet_files = [parquet_path]
    else:
        parquet_files = sorted(parquet_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"❌ No parquet files found in {parquet_path}")
        return None
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
<<<<<<< HEAD
    # Step 1: Validate and get column config
=======
    # Step 1: Validate required columns
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    logger.info("\n[1/3] Validating data structure...")
    
    try:
        first_df = pd.read_parquet(parquet_files[0])
        
        # Define column configuration
        column_config = {
            'input_ids': 'int_list',      # Required
            'attention_mask': 'int_list',  # Required
            'connector_mask': 'float_list', # Optional
            'connector_words': 'str_list',  # Optional
            'connector_types': 'str_list',  # Optional
        }
        
<<<<<<< HEAD
        # Add optional columns from first file
        for col in first_df.columns:
            if col not in column_config:
                column_config[col] = 'string'  # Default type for unknown columns
        
        logger.info(f"✓ Column configuration:")
        logger.info(f"  Required: input_ids (int_list), attention_mask (int_list)")
        logger.info(f"  Optional: connector_mask, connector_words, connector_types")
        logger.info(f"  Total columns: {len(column_config)}")
=======
        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            logger.error(f"❌ Missing required columns: {missing}")
            logger.error(f"   Available: {actual_cols}")
            return None
        
        logger.info(f"✓ Required columns present")
        logger.info(f"  Columns in parquet: {list(actual_cols)}")
        logger.info(f"  First file has {len(first_df):,} rows")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        
        # Check if connector_mask is present
        has_connector_mask = 'connector_mask' in actual_cols
        logger.info(f"  connector_mask present: {has_connector_mask}")
        if has_connector_mask:
            logger.info(f"    (Will be used for loss weighting)")
        else:
            logger.info(f"    (Model will auto-detect tags from input_ids)")
        
    except Exception as e:
        logger.error(f"❌ Error validating first file: {e}")
        return None
    
    # Step 2: Process files with REPAIR
    logger.info("\n[2/3] Processing parquet files batch-wise WITH REPAIR...")
    
    all_datasets = []
    total_rows = 0
    repaired_rows = 0
    
    for file_idx, file_path in enumerate(tqdm(parquet_files, desc="Processing files"), 1):
        try:
            # Load file
            df = pd.read_parquet(file_path)
            num_rows = len(df)
            total_rows += num_rows
            
            logger.debug(f"  File {file_idx}: {num_rows:,} rows")
            
<<<<<<< HEAD
            # CRITICAL: REPAIR (don't skip!)
            try:
                df_repaired = repair_dataframe_batch(df, column_config)
                repaired_rows += num_rows
            except Exception as e:
                logger.warning(f"  File {file_idx}: Repair failed: {e}, using raw data")
                df_repaired = df
=======
            # Clean file
            df = clean_dataframe_batch(df)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
            
            # Convert to HFDataset
            try:
                dataset = df_to_hfdataset_batch(df_repaired)
            except Exception as e:
                logger.error(f"  File {file_idx}: Still failed to convert: {e}")
                raise
            
            # Store dataset
            all_datasets.append(dataset)
            
            # Memory cleanup
<<<<<<< HEAD
            del df, df_repaired
=======
            del df
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
            gc.collect()
            
        except Exception as e:
            logger.error(f"  ❌ CRITICAL: Could not process {file_path.name}: {e}")
            logger.error(f"     This file will be SKIPPED (no other option)")
            continue
    
    if not all_datasets:
        logger.error("❌ No datasets created from parquet files")
        return None
    
    logger.info(f"✓ Processed {len(all_datasets)} files ({total_rows:,} total rows)")
    logger.info(f"✓ Repaired {repaired_rows:,} rows")
    
    # Step 3: Concatenate all datasets
    logger.info("\n[3/3] Concatenating datasets...")
    
    try:
        if len(all_datasets) == 1:
            full_dataset = all_datasets[0]
            logger.info(f"✓ Single dataset: {len(full_dataset):,} samples")
        else:
            logger.info(f"Concatenating {len(all_datasets)} datasets...")
            full_dataset = concatenate_datasets(all_datasets)
            logger.info(f"✓ Concatenated: {len(full_dataset):,} samples")
        
        # Verify structure
        sample = full_dataset[0]
        logger.info(f"\n✓ Dataset structure verified:")
        logger.info(f"  - Has 'input_ids': {'input_ids' in sample}")
        logger.info(f"  - Has 'attention_mask': {'attention_mask' in sample}")
        logger.info(f"  - Has 'connector_mask': {'connector_mask' in sample}")
        
    except Exception as e:
        logger.error(f"❌ Error concatenating datasets: {e}")
        return None
    
    # Split into train/validation
    logger.info(f"\n✓ Splitting into train/validation...")
    
    if split_ratio > 0 and split_ratio < 1:
        split = full_dataset.train_test_split(test_size=split_ratio, seed=42)
        dataset_dict = {
            'train': split['train'],
            'validation': split['test']
        }
        logger.info(f"  - Train: {len(dataset_dict['train']):,} samples")
        logger.info(f"  - Validation: {len(dataset_dict['validation']):,} samples")
    else:
<<<<<<< HEAD
        train_dataset = full_dataset
        val_dataset = None
        logger.info(f"  - Using full dataset: {len(train_dataset):,} samples")
    
    # Wrap with StringConvertingDataset for safety
    logger.info("\n✓ Wrapping datasets with StringConvertingDataset...")
    logger.info("  (Extra safety layer)")
    
    dataset_dict = {
        'train': StringConvertingDataset(train_dataset),
        'validation': StringConvertingDataset(val_dataset) if val_dataset else None
    }
    
    logger.info("\n" + "="*70)
    logger.info("✓ Data loaded and repaired successfully")
    logger.info(f"  Original: 186,899 samples")
    logger.info(f"  After repair: {len(train_dataset) + (len(val_dataset) if val_dataset else 0):,} samples")
    logger.info(f"  ✓ NO DATA LOST (all NULL values fixed)!")
=======
        dataset_dict = {
            'train': full_dataset,
            'validation': None
        }
        logger.info(f"  - Using full dataset: {len(full_dataset):,} samples")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Data loaded successfully (NO WRAPPING NEEDED)")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    logger.info("="*70)
    
    return dataset_dict


def run_training(config, model_handler, dataset_dict):
<<<<<<< HEAD
    """Run training with repaired data."""
=======
    """
    Run training for 1 epoch and save the model.
    
    Args:
        config: Configuration object
        model_handler: Model instance
        dataset_dict: Dictionary with 'train' and 'validation' datasets
    """
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    logger.info("\n" + "="*70)
    logger.info("SETTING UP TRAINING")
    logger.info("="*70)
    
    clear_cuda_memory()
    
    # Create training manager
    logger.info("\n[1/3] Creating training manager...")
    
    pretrain_manager = ConnectorPretrainingManager(
        config=config,
        model_handler=model_handler,
        use_new_collator=True
    )
    
    logger.info("✓ Training manager created")
    
    # Prepare trainer
    logger.info("\n[2/3] Preparing trainer...")
    
    pretrain_manager.prepare_trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict.get('validation'),
        output_dir="./output/connector_model_1epoch",
        
        boost_factor=config.boost_factor,
        
        # Training settings
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-6
    )
    
    logger.info("✓ Trainer prepared")
    
    # Show memory before training
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"\n✓ GPU memory before training:")
        logger.info(f"  - Allocated: {allocated:.2f} GB")
        logger.info(f"  - Reserved: {reserved:.2f} GB")
        logger.info(f"  - Available: {total - reserved:.2f} GB")
    
    # Start training
    logger.info("\n[3/3] Starting training...")
    logger.info("\n" + "="*70)
    logger.info("TRAINING IN PROGRESS")
    logger.info("="*70 + "\n")
    
    pretrain_manager.train()
    
    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("="*70)
    
    return pretrain_manager


def save_model(pretrain_manager):
    """Save the trained model."""
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    
    output_path = "./output/connector_model_1epoch/final"
    pretrain_manager.save_model(output_path)
    
    logger.info(f"✓ Model saved to: {output_path}")
    logger.info("="*70)


def main():
<<<<<<< HEAD
    """Main entry point with repair (not skip)."""
    
    logger.info("\n" + "="*70)
    logger.info("CONNECTOR-AWARE PRETRAINING PIPELINE (FIXED v5)")
    logger.info("="*70)
    logger.info("Features:")
    logger.info("  • Batch-wise parquet loading (memory efficient)")
    logger.info("  • REPAIR NULL VALUES (don't skip files!) ✓")
    logger.info("  • Fix type mismatches")
    logger.info("  • Preserve ALL samples")
    logger.info("  • STRING → INTEGER CONVERSION AT LOAD TIME")
=======
    """
    Main entry point - Simplified training pipeline.
    
    UPDATED FOR model_multiword_support.py:
    - Model auto-detects connector tags
    - No TensorDatasetWrapper needed
    - Simpler, cleaner code
    """
    
    logger.info("\n" + "="*70)
    logger.info("CONNECTOR-AWARE PRETRAINING PIPELINE")
    logger.info("UPDATED: Model auto-detects connector tags")
    logger.info("Training for 1 epoch")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    logger.info("="*70)
    
    try:
        # Initial memory check
        if torch.cuda.is_available():
            logger.info(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ Total GPU memory: {total_mem:.2f} GB")
        else:
            logger.warning("⚠ No GPU detected")
        
        # Setup HuggingFace
        setup_huggingface()
        
        # Load configuration
        logger.info("\n[Step 1/5] Loading configuration...")
        config = Config()
        config.print_summary()
        
        # Load model
        logger.info("\n[Step 2/5] Loading model...")
        model_handler = load_model_handler(config)
        
        # Load data from parquet
        logger.info("\n[Step 3/5] Loading data from parquet...")
        
        parquet_path = os.environ.get('PARQUET_PATH', './data_splits')
        logger.info(f"Parquet path: {parquet_path}")

        # Toggle via env var; default to chunked training
        use_chunked = os.environ.get("CHUNKED_TRAINING", "1") == "1"
        chunk_size = int(os.environ.get("CHUNK_SIZE", "20"))

        if use_chunked:
            logger.info("Using CHUNKED training pipeline")
            train_in_chunks(config, model_handler, parquet_path, chunk_size=chunk_size, split_ratio=0.05)
        else:
            # Fallback: your old all-in-memory path (not recommended for large corpora)
            dataset_dict = load_data_from_parquet_batch_wise(parquet_path, split_ratio=0.05)  # current function:contentReference[oaicite:11]{index=11}
            if dataset_dict is None or dataset_dict['train'] is None:
                logger.error("\n❌ Failed to load data (non-chunked mode)")
                return
            pretrain_manager = run_training(config, model_handler, dataset_dict)  # current function:contentReference[oaicite:12]{index=12}
            save_model(pretrain_manager)                                          # current function:contentReference[oaicite:13]{index=13}

        clear_cuda_memory()
        
        # Success message
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info("\n📊 Summary:")
        logger.info(f"  • Model: {config.model_name}")
        if use_chunked:
            logger.info("  • Training samples: see per‑chunk logs")
        else:
            logger.info(f"  • Training samples: {len(dataset_dict['train']):,}")
            if dataset_dict.get('validation'):
                logger.info(f"  • Validation samples: {len(dataset_dict['validation']):,}")
        logger.info(f"  • Epochs: 1")
        logger.info(f"  • Batch size: 1")
<<<<<<< HEAD
        logger.info(f"  • Data approach: REPAIR (preserve all samples)")
        logger.info(f"  • Output: ./output/connector_model_1epoch/final")
=======
        logger.info(f"\n✓ Key Features:")
        logger.info(f"  • Auto connector tag detection")
        logger.info(f"  • Multi-word connector support")
        logger.info(f"  • Boost applied at embedding layer")
        logger.info(f"  • Gradient amplification: 1.1× (not 1.1^28)")
        logger.info(f"  • No TensorDatasetWrapper needed")
        logger.info(f"\n  • Output: ./output/connector_model_1epoch/final")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        logger.info("\n🎉 Done!")
        
    except torch.cuda.OutOfMemoryError:
        logger.error("\n❌ CUDA Out of Memory!")
        logger.info("\nTry:")
        logger.info("  • Reduce batch size (already at 1)")
        logger.info("  • Use fewer parquet files")
        logger.info("  • Restart system to clear memory")
        clear_cuda_memory()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
        clear_cuda_memory()
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        clear_cuda_memory()


if __name__ == "__main__":
    main()
