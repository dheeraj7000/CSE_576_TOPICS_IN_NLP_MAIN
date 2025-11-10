#!/usr/bin/env python3
"""
main.py - CLEAN VERSION FOR LLAMA 3.2 3B TRAINING

Simple entry point for connector-aware pretraining.
"""

import os
import gc
import torch
import logging
from pathlib import Path
from datasets import load_from_disk, concatenate_datasets

from utils.config import Config
from pretrain.model import initialize_model
from pretrain.trainer import ConnectorPretrainingManager
from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation, DirectParquetDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_cuda_memory():
    """Clear CUDA memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    logger.info("✓ GPU memory cleared")


def load_model_handler(config: Config):
    """Load and prepare model."""
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


def load_data_from_parquet(parquet_path: str, split_ratio: float = 0.05):
    """Load data from parquet files."""
    logger.info("\n" + "="*70)
    logger.info("LOADING DATA FROM PARQUET")
    logger.info("="*70)
    
    parquet_path = Path(parquet_path)
    
    if not parquet_path.exists():
        logger.error(f"❌ Path not found: {parquet_path}")
        return None
    
    # Find parquet files
    if parquet_path.is_file():
        parquet_files = [parquet_path]
    else:
        parquet_files = sorted(parquet_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"❌ No parquet files found in {parquet_path}")
        return None
    
    logger.info(f"✓ Found {len(parquet_files)} parquet files")
    
    # Load dataset
    try:
        dataset = DirectParquetDataset(parquet_path)
        logger.info(f"✓ Loaded {len(dataset):,} samples")
        
        # Split into train/validation
        indices = list(range(len(dataset)))
        split_idx = int(len(dataset) * (1 - split_ratio))
        
        # Note: For HF Trainer, we need to convert to HF Dataset format
        # This is a simple dataset for training
        dataset_dict = {
            'train': dataset,
            'validation': None
        }
        
        logger.info(f"✓ Train: {split_idx:,} samples")
        if split_ratio > 0:
            logger.info(f"✓ Validation: {len(dataset) - split_idx:,} samples")
        
        logger.info("="*70)
        return dataset_dict
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None


def run_training(config, model_handler, dataset_dict):
    """Run training."""
    logger.info("\n" + "="*70)
    logger.info("SETTING UP TRAINING")
    logger.info("="*70)
    
    clear_cuda_memory()
    
    # Create training manager
    logger.info("Creating training manager...")
    pretrain_manager = ConnectorPretrainingManager(
        config=config,
        model_handler=model_handler,
        use_new_collator=True
    )
    logger.info("✓ Training manager created")
    
    # Prepare trainer
    logger.info("Preparing trainer...")
    pretrain_manager.prepare_trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict.get('validation'),
        output_dir="./output/connector_model",
        boost_factor=config.boost_factor,
        num_epochs=config.num_train_epochs,
        batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate
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
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
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
    
    output_path = "./output/connector_model/final"
    pretrain_manager.save_model(output_path)
    
    logger.info(f"✓ Model saved to: {output_path}")
    logger.info("="*70)


def main():
    """Main entry point."""
    logger.info("\n" + "="*70)
    logger.info("CONNECTOR-AWARE PRETRAINING PIPELINE")
    logger.info("="*70)
    
    try:
        # Initial memory check
        if torch.cuda.is_available():
            logger.info(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ Total GPU memory: {total_mem:.2f} GB")
        else:
            logger.warning("⚠ No GPU detected - using CPU (slow!)")
        
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
        
        dataset_dict = load_data_from_parquet(parquet_path, split_ratio=0.05)
        
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
        logger.info(f"  • Epochs: {config.num_train_epochs}")
        logger.info(f"  • Output: ./output/connector_model/final")
        logger.info("\n🎉 Done!")
        
    except torch.cuda.OutOfMemoryError:
        logger.error("\n❌ CUDA Out of Memory!")
        logger.info("\nTry:")
        logger.info("  • Reduce batch size")
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
