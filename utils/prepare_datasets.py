#!/usr/bin/env python3
"""
prepare_datasets.py - FIXED: Memory-Optimized Dataset Preparation

Downloads and prepares 4 datasets for discourse-aware reasoning pretraining:
- ArXiv (scientific papers)
- PubMed (biomedical papers)
- Legal (legal documents)
- OpenWebMath (mathematical text)

Memory-optimized: Saves every 10,000 documents to prevent OOM.
Each batch saved to separate file, then combined.

Usage:
    python prepare_datasets.py arxiv          # Download ArXiv
    python prepare_datasets.py pubmed         # Download PubMed
    python prepare_datasets.py legal          # Download Legal
    python prepare_datasets.py openwebmath    # Download OpenWebMath
    python prepare_datasets.py combine        # Combine all
    python prepare_datasets.py status         # Check status
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List
import warnings
import shutil

from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    Dataset,
    DownloadConfig,
)

from config import DATASETS, COMBINED_DATASET_PATH, BATCH_SIZE, NUM_WORKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['REQUESTS_TIMEOUT'] = '3600'
warnings.filterwarnings('ignore')


# ============================================================================
# DATASET PREPARATION
# ============================================================================

class DatasetPreparer:
    """Handles downloading and preparing datasets with domain labels."""
    
    def __init__(self, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
        """Initialize dataset preparer."""
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets_dir = Path(COMBINED_DATASET_PATH).parent / "raw_datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Datasets directory: {self.datasets_dir}")
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get path for dataset."""
        return self.datasets_dir / dataset_name
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if dataset already downloaded."""
        path = self.get_dataset_path(dataset_name)
        return path.exists() and len(os.listdir(path)) > 0
    
    def download_arxiv(self) -> Dataset:
        """Download ArXiv dataset with memory optimization."""
        dataset_name = 'arxiv'
        save_path = self.get_dataset_path(dataset_name)
        
        if self.dataset_exists(dataset_name):
            logger.info(f"✓ ArXiv already downloaded at {save_path}")
            return load_from_disk(str(save_path))
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING ARXIV (MEMORY-OPTIMIZED)")
        logger.info("="*80 + "\n")
        
        try:
            # Load with streaming (no num_proc for streaming!)
            stream = load_dataset(
                "armanc/scientific_papers",
                "arxiv",
                split="train",
                streaming=True,
                download_config=DownloadConfig(resume_download=True)
            )
            
            batch_data = []
            batch_num = 0
            total_count = 0
            batch_files = []
            
            logger.info(f"Processing in batches of {self.batch_size}...")
            
            for idx, example in enumerate(stream):
                example['domain'] = dataset_name
                batch_data.append(example)
                total_count += 1
                
                # Save batch
                if len(batch_data) >= self.batch_size:
                    batch_file = save_path / f"batch_{batch_num:04d}"
                    logger.info(f"  Saving batch {batch_num}: papers {total_count-self.batch_size+1}-{total_count}")
                    
                    batch_dataset = Dataset.from_dict(
                        {k: [item[k] for item in batch_data] for k in batch_data[0].keys()}
                    )
                    batch_dataset.save_to_disk(str(batch_file))
                    batch_files.append(batch_file)
                    
                    batch_data = []
                    batch_num += 1
            
            # Save final batch
            if batch_data:
                batch_file = save_path / f"batch_{batch_num:04d}"
                logger.info(f"  Saving final batch {batch_num}: {len(batch_data)} papers")
                batch_dataset = Dataset.from_dict(
                    {k: [item[k] for item in batch_data] for k in batch_data[0].keys()}
                )
                batch_dataset.save_to_disk(str(batch_file))
                batch_files.append(batch_file)
            
            # Combine all batches
            logger.info(f"\n  Combining {len(batch_files)} batches...")
            all_datasets = [load_from_disk(str(f)) for f in batch_files]
            combined = concatenate_datasets(all_datasets)
            
            # Save combined
            # Clear existing if needed
            if save_path.exists():
                shutil.rmtree(save_path)
            
            combined.save_to_disk(str(save_path))
            
            # Clean up batch files
            for batch_file in batch_files:
                if batch_file.exists():
                    shutil.rmtree(batch_file)
            
            logger.info(f"✓ ArXiv downloaded: {len(combined)} papers")
            return combined
            
        except Exception as e:
            logger.error(f"✗ Error downloading ArXiv: {e}")
            raise
    
    def download_pubmed(self) -> Dataset:
        """Download PubMed dataset."""
        dataset_name = 'pubmed'
        save_path = self.get_dataset_path(dataset_name)
        
        if self.dataset_exists(dataset_name):
            logger.info(f"✓ PubMed already downloaded at {save_path}")
            return load_from_disk(str(save_path))
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING PUBMED")
        logger.info("="*80 + "\n")
        
        try:
            stream = load_dataset(
                "armanc/scientific_papers",
                "pubmed",
                split="train",
                streaming=True,
                download_config=DownloadConfig(resume_download=True)
            )
            
            batch_data = []
            batch_num = 0
            total_count = 0
            batch_files = []
            
            logger.info(f"Processing in batches of {self.batch_size}...")
            
            for idx, example in enumerate(stream):
                example['domain'] = dataset_name
                batch_data.append(example)
                total_count += 1
                
                if len(batch_data) >= self.batch_size:
                    batch_file = save_path / f"batch_{batch_num:04d}"
                    logger.info(f"  Saving batch {batch_num}: papers {total_count-self.batch_size+1}-{total_count}")
                    batch_dataset = Dataset.from_dict(
                        {k: [item[k] for item in batch_data] for k in batch_data[0].keys()}
                    )
                    batch_dataset.save_to_disk(str(batch_file))
                    batch_files.append(batch_file)
                    batch_data = []
                    batch_num += 1
            
            if batch_data:
                batch_file = save_path / f"batch_{batch_num:04d}"
                logger.info(f"  Saving final batch {batch_num}: {len(batch_data)} papers")
                batch_dataset = Dataset.from_dict(
                    {k: [item[k] for item in batch_data] for k in batch_data[0].keys()}
                )
                batch_dataset.save_to_disk(str(batch_file))
                batch_files.append(batch_file)
            
            logger.info(f"\n  Combining {len(batch_files)} batches...")
            all_datasets = [load_from_disk(str(f)) for f in batch_files]
            combined = concatenate_datasets(all_datasets)
            
            if save_path.exists():
                shutil.rmtree(save_path)
            combined.save_to_disk(str(save_path))
            
            for batch_file in batch_files:
                if batch_file.exists():
                    shutil.rmtree(batch_file)
            
            logger.info(f"✓ PubMed downloaded: {len(combined)} papers")
            return combined
            
        except Exception as e:
            logger.error(f"✗ Error downloading PubMed: {e}")
            raise
    
    def download_legal(self) -> Dataset:
        """Download Legal dataset."""
        dataset_name = 'legal'
        save_path = self.get_dataset_path(dataset_name)
        
        if self.dataset_exists(dataset_name):
            logger.info(f"✓ Legal already downloaded at {save_path}")
            return load_from_disk(str(save_path))
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING LEGAL")
        logger.info("="*80 + "\n")
        
        try:
            try:
                dataset = load_dataset(
                    "pile-of-law/pile-of-law",
                    "r_legaladvice",
                    split="train",
                    download_config=DownloadConfig(resume_download=True)
                )
            except:
                logger.warning("Primary legal dataset unavailable, trying alternative...")
                dataset = load_dataset("jonathanli/pile-of-law-sample", split="train")
            
            dataset = dataset.map(lambda x: {**x, 'domain': dataset_name})
            
            save_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_path))
            logger.info(f"✓ Legal downloaded: {len(dataset)} documents")
            return dataset
            
        except Exception as e:
            logger.error(f"✗ Error downloading Legal: {e}")
            raise
    
    def download_openwebmath(self) -> Dataset:
        """Download OpenWebMath dataset."""
        dataset_name = 'openwebmath'
        save_path = self.get_dataset_path(dataset_name)
        
        if self.dataset_exists(dataset_name):
            logger.info(f"✓ OpenWebMath already downloaded at {save_path}")
            return load_from_disk(str(save_path))
        
        logger.info("\n" + "="*80)
        logger.info("DOWNLOADING OPENWEBMATH")
        logger.info("="*80 + "\n")
        
        try:
            dataset = load_dataset(
                "open-web-math/open-web-math",
                split="train",
                download_config=DownloadConfig(resume_download=True)
            )
            
            dataset = dataset.map(lambda x: {**x, 'domain': dataset_name})
            
            save_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_path))
            logger.info(f"✓ OpenWebMath downloaded: {len(dataset)} documents")
            return dataset
            
        except Exception as e:
            logger.error(f"✗ Error downloading OpenWebMath: {e}")
            raise
    
    def combine_datasets(self) -> Dataset:
        """Combine all downloaded datasets."""
        
        logger.info("\n" + "="*80)
        logger.info("COMBINING DATASETS")
        logger.info("="*80 + "\n")
        
        combined_path = Path(COMBINED_DATASET_PATH)
        combined_path.mkdir(parents=True, exist_ok=True)
        
        all_datasets = []
        
        for dataset_name in ['arxiv', 'pubmed', 'legal', 'openwebmath']:
            path = self.get_dataset_path(dataset_name)
            
            if not path.exists():
                logger.warning(f"⚠ {dataset_name} not found at {path}, skipping...")
                continue
            
            try:
                dataset = load_from_disk(str(path))
                
                if 'domain' not in dataset.column_names:
                    dataset = dataset.map(lambda x: {**x, 'domain': dataset_name})
                
                all_datasets.append(dataset)
                logger.info(f"✓ Loaded {dataset_name}: {len(dataset)} examples")
                
            except Exception as e:
                logger.error(f"✗ Error loading {dataset_name}: {e}")
                continue
        
        if not all_datasets:
            logger.error("✗ No datasets found to combine!")
            return None
        
        logger.info(f"\nCombining {len(all_datasets)} datasets...")
        combined = concatenate_datasets(all_datasets)
        
        # Clear existing combined dataset if it exists
        if combined_path.exists():
            shutil.rmtree(combined_path)
        
        combined.save_to_disk(str(combined_path))
        
        logger.info(f"✓ Combined dataset saved: {len(combined)} total examples")
        logger.info(f"  Location: {combined_path}")
        
        # Domain distribution
        logger.info("\nDomain distribution:")
        domains = combined.unique('domain')
        for domain in domains:
            count = len(combined.filter(lambda x: x['domain'] == domain))
            pct = (count / len(combined)) * 100
            logger.info(f"  - {domain}: {count:,} ({pct:.1f}%)")
        
        return combined
    
    def status(self) -> None:
        """Show download status."""
        logger.info("\n" + "="*80)
        logger.info("DATASET STATUS")
        logger.info("="*80 + "\n")
        
        for dataset_name in ['arxiv', 'pubmed', 'legal', 'openwebmath']:
            path = self.get_dataset_path(dataset_name)
            if self.dataset_exists(dataset_name):
                try:
                    dataset = load_from_disk(str(path))
                    logger.info(f"✓ {dataset_name.upper()}: {len(dataset):,} examples")
                except:
                    logger.info(f"⚠ {dataset_name.upper()}: directory exists but corrupted")
            else:
                logger.info(f"✗ {dataset_name.upper()}: not downloaded")
        
        combined_path = Path(COMBINED_DATASET_PATH)
        if combined_path.exists():
            try:
                combined = load_from_disk(str(combined_path))
                logger.info(f"\n✓ COMBINED: {len(combined):,} total examples")
            except:
                logger.info(f"\n⚠ COMBINED: directory exists but corrupted")
        else:
            logger.info(f"\n✗ COMBINED: not created")
        
        logger.info("\n" + "="*80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python prepare_datasets.py arxiv          # Download ArXiv")
        print("  python prepare_datasets.py pubmed         # Download PubMed")
        print("  python prepare_datasets.py legal          # Download Legal")
        print("  python prepare_datasets.py openwebmath    # Download OpenWebMath")
        print("  python prepare_datasets.py combine        # Combine all datasets")
        print("  python prepare_datasets.py status         # Show download status\n")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    preparer = DatasetPreparer()
    
    try:
        if command == 'arxiv':
            preparer.download_arxiv()
        elif command == 'pubmed':
            preparer.download_pubmed()
        elif command == 'legal':
            preparer.download_legal()
        elif command == 'openwebmath':
            preparer.download_openwebmath()
        elif command == 'combine':
            preparer.combine_datasets()
        elif command == 'status':
            preparer.status()
        else:
            logger.error(f"Unknown command: {command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()