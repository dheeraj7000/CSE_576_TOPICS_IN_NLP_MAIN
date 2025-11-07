#!/usr/bin/env python3
"""
main.py - WITH data_loader.py COMPATIBILITY
Converts HFDataset to return torch tensors so data_loader.py collator works

Key fix: TensorDatasetWrapper converts list columns to torch.Tensor
This way data_loader.py's ConnectorDataCollatorWithMaskCreation can process them
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

# Import project modules
from utils.config import Config
from pretrain.model import initialize_model
from pretrain.trainer import ConnectorPretrainingManager
from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation

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
    """
    Load and prepare model with connector tokens.
    
    Args:
        config: Configuration object
    Returns:
        model_handler: Initialized Model instance
    """
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


def clean_dataframe_batch(df):
    """
    Clean a single batch (from one file or chunk).
    
    Fixes:
    - Converts numpy arrays to lists
    - Handles null values in list columns
    - Ensures consistent data types
    
    Args:
        df: Pandas DataFrame (single file/batch)
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
    # Process each column
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == object:
            # Check if column contains numpy arrays
            sample = cleaned_df[col].iloc[0]
            if isinstance(sample, np.ndarray):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            # Handle null values in list columns
            elif isinstance(sample, list):
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: x if (isinstance(x, list) and x is not None) else []
                )
    
    return cleaned_df


def df_to_hfdataset_batch(df):
    """
    Convert a single batch DataFrame to HFDataset.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        HFDataset
    """
    # Convert to dict format, ensuring all values are Python types (not numpy)
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].tolist()
    
    dataset = HFDataset.from_dict(data_dict)
    return dataset


class TensorDatasetWrapper:
    """
    Wrapper that converts HFDataset items to torch tensors.
    
    CRITICAL: data_loader.py's CollectorDataCollatorWithMaskCreation expects
    batch items to have torch.Tensor values, not Python lists.
    
    This wrapper ensures:
    - input_ids → torch.Tensor (torch.long)
    - attention_mask → torch.Tensor (torch.long)
    - connector_mask → torch.Tensor (torch.float) if present
    """
    
    def __init__(self, dataset: HFDataset):
        self.dataset = dataset
        self.length = len(dataset)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Convert input_ids to tensor
        input_ids = item.get("input_ids")
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Convert attention_mask to tensor
        attention_mask = item.get("attention_mask")
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # Convert connector_mask to tensor if present
        connector_mask = item.get("connector_mask")
        if connector_mask is not None:
            if isinstance(connector_mask, list):
                connector_mask = torch.tensor(connector_mask, dtype=torch.float)
            else:
                connector_mask = torch.tensor(connector_mask, dtype=torch.float)
        
        # Return dict with tensor values
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "connector_mask": connector_mask if connector_mask is not None else torch.ones_like(input_ids, dtype=torch.float),
            "connector_words": item.get("connector_words", []),
            "connector_types": item.get("connector_types", [])
        }


def load_data_from_parquet_batch_wise(parquet_path: str, split_ratio: float = 0.05):
    """
    Load data from parquet files BATCH-WISE (memory efficient).
    Wraps with TensorDatasetWrapper for data_loader.py compatibility.
    
    Process:
    1. Find all parquet files
    2. Load ONE file at a time
    3. Clean and convert each file to HFDataset
    4. Concatenate all datasets
    5. WRAP with TensorDatasetWrapper
    6. Split into train/validation
    
    Args:
        parquet_path: Path to parquet directory or file
        split_ratio: Ratio for train/validation split
    
    Returns:
        dataset_dict: Dictionary with 'train' and 'validation' keys (wrapped)
    """
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA FROM PARQUET (BATCH-WISE)")
    logger.info("With TensorDatasetWrapper for data_loader.py compatibility")
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
    logger.info("Processing files one at a time (batch-wise)...")
    
    # Step 1: Validate required columns (check first file only)
    logger.info("\n[1/3] Validating data structure (checking first file)...")
    
    try:
        first_df = pd.read_parquet(parquet_files[0])
        
        required_cols = {'input_ids', 'attention_mask'}
        actual_cols = set(first_df.columns)
        
        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            logger.error(f"❌ Missing required columns: {missing}")
            return None
        
        logger.info(f"✓ All required columns present")
        logger.info(f"  Columns: {list(first_df.columns)}")
        logger.info(f"  First file has {len(first_df):,} rows")
        
    except Exception as e:
        logger.error(f"❌ Error validating first file: {e}")
        return None
    
    # Step 2: Process files one by one
    logger.info("\n[2/3] Processing parquet files batch-wise...")
    
    all_datasets = []
    total_rows = 0
    
    for file_idx, file_path in enumerate(tqdm(parquet_files, desc="Processing files"), 1):
        try:
            # Load file
            df = pd.read_parquet(file_path)
            num_rows = len(df)
            total_rows += num_rows
            
            logger.debug(f"  File {file_idx}: {num_rows:,} rows")
            
            # Clean file (batch-wise)
            df = clean_dataframe_batch(df)
            
            # Convert to HFDataset
            dataset = df_to_hfdataset_batch(df)
            
            # Store dataset
            all_datasets.append(dataset)
            
            # Memory cleanup after each file
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            continue
    
    if not all_datasets:
        logger.error("❌ No datasets created from parquet files")
        return None
    
    logger.info(f"✓ Processed {len(all_datasets)} files ({total_rows:,} total rows)")
    
    # Step 3: Concatenate all datasets
    logger.info("\n[3/3] Concatenating and wrapping datasets...")
    
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
    
    # Split into train/validation BEFORE wrapping
    logger.info(f"\n✓ Splitting into train/validation...")
    
    if split_ratio > 0 and split_ratio < 1:
        split = full_dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split['train']
        val_dataset = split['test']
        logger.info(f"  - Train: {len(train_dataset):,} samples")
        logger.info(f"  - Validation: {len(val_dataset):,} samples")
    else:
        train_dataset = full_dataset
        val_dataset = None
        logger.info(f"  - Using full dataset: {len(train_dataset):,} samples")
    
    # NOW wrap with TensorDatasetWrapper
    logger.info("\n✓ Wrapping datasets with TensorDatasetWrapper...")
    logger.info("  (Converting lists to torch.Tensor for data_loader.py)")
    
    dataset_dict = {
        'train': TensorDatasetWrapper(train_dataset),
        'validation': TensorDatasetWrapper(val_dataset) if val_dataset else None
    }
    
    logger.info("\n" + "="*70)
    logger.info("✓ Data loaded and wrapped successfully")
    logger.info("="*70)
    
    return dataset_dict


def run_training(config, model_handler, dataset_dict):
    """
    Run training for 1 epoch and save the model.
    Uses data_loader.py's ConnectorDataCollatorWithMaskCreation
    
    Args:
        config: Configuration object
        model_handler: Model instance
        dataset_dict: Dictionary with wrapped 'train' and 'validation' datasets
    """
    logger.info("\n" + "="*70)
    logger.info("SETTING UP TRAINING")
    logger.info("="*70)
    
    clear_cuda_memory()
    
    # Create training manager
    logger.info("\n[1/3] Creating training manager...")
    
    pretrain_manager = ConnectorPretrainingManager(
        config=config,
        model_handler=model_handler,
        use_new_collator=True  # Uses data_loader.py collator
    )
    
    logger.info("✓ Training manager created")
    
    # Prepare trainer with data_loader.py collator
    logger.info("\n[2/3] Preparing trainer with data_loader.py collator...")
    
    pretrain_manager.prepare_trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict.get('validation'),
        output_dir="./output/connector_model_1epoch",
        
        # Connector boosting settings
        boost_factor=config.boost_factor,
        use_amplification=False,
        amplification_strength=1.0,
        
        # Training settings
        num_epochs=1,
        batch_size=1,
        learning_rate=5e-6
    )
    
    logger.info("✓ Trainer prepared with data_loader.py")
    
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
    """
    Save the trained model.
    
    Args:
        pretrain_manager: Training manager instance
    """
    logger.info("\n" + "="*70)
    logger.info("SAVING MODEL")
    logger.info("="*70)
    
    output_path = "./output/connector_model_1epoch/final"
    pretrain_manager.save_model(output_path)
    
    logger.info(f"✓ Model saved to: {output_path}")
    logger.info("="*70)


def main():
    """
    Main entry point - Runs full training pipeline with data_loader.py integration.
    """
    
    logger.info("\n" + "="*70)
    logger.info("CONNECTOR-AWARE PRETRAINING PIPELINE")
    logger.info("BATCH-WISE PARQUET LOADING WITH data_loader.py")
    logger.info("Training for 1 epoch")
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
        
        # Load data from parquet (batch-wise with wrapping)
        logger.info("\n[Step 3/5] Loading data from parquet...")
        
        parquet_path = os.environ.get('PARQUET_PATH', './data_splits')
        logger.info(f"Parquet path: {parquet_path}")
        
        dataset_dict = load_data_from_parquet_batch_wise(parquet_path, split_ratio=0.05)
        
        if dataset_dict is None or dataset_dict['train'] is None:
            logger.error("\n❌ Failed to load data")
            logger.info("\n💡 Make sure:")
            logger.info("  • Parquet files exist at: ./data_splits")
            logger.info("  • Or set: export PARQUET_PATH=/path/to/parquet")
            logger.info("  • Parquet columns include: input_ids, attention_mask")
            return
        
        # Train model
        logger.info("\n[Step 4/5] Training model...")
        pretrain_manager = run_training(config, model_handler, dataset_dict)
        
        # Save model
        logger.info("\n[Step 5/5] Saving model...")
        save_model(pretrain_manager)
        
        # Cleanup
        clear_cuda_memory()
        
        # Success message
        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info("\n📊 Summary:")
        logger.info(f"  • Model: {config.model_name}")
        logger.info(f"  • Training samples: {len(dataset_dict['train']):,}")
        if dataset_dict.get('validation'):
            logger.info(f"  • Validation samples: {len(dataset_dict['validation']):,}")
        logger.info(f"  • Epochs: 1")
        logger.info(f"  • Batch size: 1")
        logger.info(f"  • Loading mode: Batch-wise (memory efficient)")
        logger.info(f"  • Data collator: data_loader.py (ConnectorDataCollatorWithMaskCreation)")
        logger.info(f"  • Output: ./output/connector_model_1epoch/final")
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