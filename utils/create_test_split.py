import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_train_test_split(
    checkpoint_dir: str = "./dataset",
    test_split: float = 0.05,
    output_dir: str = "./data_splits"
):
    """
    Split preprocessed checkpoints into train/test using chunked writing.
    """
    logger.info("="*80)
    logger.info("CREATING TRAIN/TEST SPLIT (chunked/write-per-file)")
    logger.info("="*80)

    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_files = sorted(checkpoint_path.glob("checkpoint_*.parquet"))
    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_dir}")
        return

    logger.info(f"\nFound {len(checkpoint_files)} checkpoint files")
    train_paths = []
    test_paths = []
    total_train = 0
    total_test = 0

    for idx, file in enumerate(tqdm(checkpoint_files, desc="Processing checkpoints")):
        df = pd.read_parquet(file)
        # Shuffle within file
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_split))

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        total_train += len(train_df)
        total_test += len(test_df)

        train_path = output_path / f'train_chunk_{idx:03d}.parquet'
        test_path = output_path / f'test_chunk_{idx:03d}.parquet'
        train_df.to_parquet(train_path)
        test_df.to_parquet(test_path)
        train_paths.append(str(train_path))
        test_paths.append(str(test_path))

    logger.info(f"✓ Chunked train splits: {len(train_paths)} files ({total_train:,} total rows)")
    logger.info(f"✓ Chunked test splits: {len(test_paths)} files ({total_test:,} total rows)")
    logger.info(f"\nAll splits saved in directory: {output_path}")
    logger.info("\n" + "="*80)
    logger.info("✓ CHUNKED SPLIT COMPLETE")
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser(description="Create train/test split from preprocessed data (chunked writing)")
    parser.add_argument("--checkpoint_dir", type=str, default="./dataset", help="Directory with checkpoint_*.parquet files")
    parser.add_argument("--test_split", type=float, default=0.05, help="Fraction for test set (default: 0.05 = 5%)")
    parser.add_argument("--output_dir", type=str, default="./data_splits", help="Output directory for train/test files")
    args = parser.parse_args()
    create_train_test_split(
        checkpoint_dir=args.checkpoint_dir,
        test_split=args.test_split,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
