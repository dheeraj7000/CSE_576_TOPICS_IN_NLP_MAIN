#!/usr/bin/env python3
"""
main.py - UPDATED FOR model_multiword_support.py

CRITICAL SIMPLIFICATION:
- Model auto-detects connector tags from input_ids
- NO TensorDatasetWrapper needed!
- Simpler data pipeline
- Works with or without connector_mask in parquet

FLOW:
parquet → input_ids + attention_mask → model detects tags → boost applied
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
    Clean a single batch DataFrame.
    
    Converts numpy arrays to lists for HFDataset compatibility.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    cleaned_df = df.copy()
    
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
    
    return cleaned_df


def df_to_hfdataset_batch(df):
    """
    Convert DataFrame to HFDataset.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        HFDataset
    """
    # Convert to dict format
    data_dict = {}
    for col in df.columns:
        data_dict[col] = df[col].tolist()
    
    dataset = HFDataset.from_dict(data_dict)
    return dataset


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
    
    # Step 1: Validate required columns
    logger.info("\n[1/3] Validating data structure...")
    
    try:
        first_df = pd.read_parquet(parquet_files[0])
        
        required_cols = {'input_ids', 'attention_mask'}
        actual_cols = set(first_df.columns)
        
        if not required_cols.issubset(actual_cols):
            missing = required_cols - actual_cols
            logger.error(f"❌ Missing required columns: {missing}")
            logger.error(f"   Available: {actual_cols}")
            return None
        
        logger.info(f"✓ Required columns present")
        logger.info(f"  Columns in parquet: {list(actual_cols)}")
        logger.info(f"  First file has {len(first_df):,} rows")
        
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
            
            # Clean file
            df = clean_dataframe_batch(df)
            
            # Convert to HFDataset
            dataset = df_to_hfdataset_batch(df)
            
            # Store dataset
            all_datasets.append(dataset)
            
            # Memory cleanup
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
        dataset_dict = {
            'train': full_dataset,
            'validation': None
        }
        logger.info(f"  - Using full dataset: {len(full_dataset):,} samples")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Data loaded successfully (NO WRAPPING NEEDED)")
    logger.info("="*70)
    
    return dataset_dict


def run_training(config, model_handler, dataset_dict):
    """
    Run training for 1 epoch and save the model.
    
    Args:
        config: Configuration object
        model_handler: Model instance
        dataset_dict: Dictionary with 'train' and 'validation' datasets
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
    
    # Prepare trainer
    logger.info("\n[2/3] Preparing trainer...")
    
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
        
        dataset_dict = load_data_from_parquet_batch_wise(parquet_path, split_ratio=0.05)
        
        if dataset_dict is None or dataset_dict['train'] is None:
            logger.error("\n❌ Failed to load data")
            logger.info("\n💡 Make sure:")
            logger.info("  • Parquet files exist at: ./data_splits")
            logger.info("  • Or set: export PARQUET_PATH=/path/to/parquet")
            logger.info("  • Parquet has columns: input_ids, attention_mask")
            logger.info("  • connector_mask is optional (model auto-detects)")
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
        logger.info(f"\n✓ Key Features:")
        logger.info(f"  • Auto connector tag detection")
        logger.info(f"  • Multi-word connector support")
        logger.info(f"  • Boost applied at embedding layer")
        logger.info(f"  • Gradient amplification: 1.1× (not 1.1^28)")
        logger.info(f"  • No TensorDatasetWrapper needed")
        logger.info(f"\n  • Output: ./output/connector_model_1epoch/final")
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
