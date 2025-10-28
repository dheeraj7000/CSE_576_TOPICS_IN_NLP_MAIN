"""
Usage: python download_memory_optimized.py arxiv
"""

from datasets import load_dataset, concatenate_datasets, load_from_disk, Dataset, DownloadConfig
import sys
import os
import warnings

os.environ['REQUESTS_TIMEOUT'] = '3600'
warnings.filterwarnings('ignore')

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def download_arxiv_streaming_batched():
    """Download ArXiv using streaming with batched saving (prevents OOM)."""
    print("\n" + "="*80)
    print("DOWNLOADING ARXIV (MEMORY-OPTIMIZED)")
    print("="*80)
    print("Strategy: Download + Save every 10K papers to avoid memory issues")
    print("="*80 + "\n")
    
    # Load as stream
    print("Starting streaming download...")
    stream = load_dataset(
        "armanc/scientific_papers",
        "arxiv",
        split="train",
        streaming=True,
        download_config=DownloadConfig(resume_download=True)
    )
    
    # Process in batches
    BATCH_SIZE = 10000
    batch_data = []
    batch_num = 0
    total_count = 0
    
    print(f"Processing in batches of {BATCH_SIZE:,} papers...")
    print("(This prevents memory overflow)\n")
    
    for item in stream:
        batch_data.append(item)
        total_count += 1
        
        # Progress update
        if total_count % 1000 == 0:
            print(f"  Downloaded: {total_count:,} papers...", end='\r')
        
        # Save batch when full
        if len(batch_data) >= BATCH_SIZE:
            print(f"\n  Saving batch {batch_num} ({BATCH_SIZE:,} papers)...")
            
            # Convert batch to dataset
            batch_dataset = Dataset.from_dict({
                key: [item[key] for item in batch_data]
                for key in batch_data[0].keys()
            })
            
            # Add domain label
            batch_dataset = batch_dataset.map(
                lambda x: {**x, 'domain': 'arxiv'},
                desc=f"Labeling batch {batch_num}"
            )
            
            # Save batch
            batch_path = f"./arxiv_batch_{batch_num:03d}"
            batch_dataset.save_to_disk(batch_path)
            print(f"  ✓ Batch {batch_num} saved to {batch_path}")
            
            # Clear memory
            batch_data = []
            batch_num += 1
    
    # Save remaining data
    if batch_data:
        print(f"\n  Saving final batch {batch_num} ({len(batch_data):,} papers)...")
        batch_dataset = Dataset.from_dict({
            key: [item[key] for item in batch_data]
            for key in batch_data[0].keys()
        })
        batch_dataset = batch_dataset.map(
            lambda x: {**x, 'domain': 'arxiv'},
            desc=f"Labeling batch {batch_num}"
        )
        batch_path = f"./arxiv_batch_{batch_num:03d}"
        batch_dataset.save_to_disk(batch_path)
        print(f"  ✓ Batch {batch_num} saved")
    
    print(f"\n✓ Downloaded {total_count:,} papers in {batch_num + 1} batches")
    
    # Combine batches
    print("\nCombining all batches...")
    batch_datasets = []
    for i in range(batch_num + 1):
        batch_path = f"./arxiv_batch_{i:03d}"
        if os.path.exists(batch_path):
            batch_datasets.append(load_from_disk(batch_path))
            print(f"  Loaded batch {i}")
    
    combined = concatenate_datasets(batch_datasets)
    print(f"✓ Combined: {len(combined):,} papers")
    
    # Save final dataset
    print("\nSaving final ArXiv dataset...")
    combined.save_to_disk("./arxiv_dataset")
    
    # Clean up batch files
    print("Cleaning up temporary batch files...")
    for i in range(batch_num + 1):
        import shutil
        batch_path = f"./arxiv_batch_{i:03d}"
        if os.path.exists(batch_path):
            shutil.rmtree(batch_path)
    
    print("\n✓ ArXiv saved to: ./arxiv_dataset")
    print(f"✓ Total papers: {len(combined):,}")
    print("\nNext: Run: python download_memory_optimized.py pubmed")


def download_pubmed_streaming_batched():
    """Download PubMed using streaming with batched saving."""
    print("\n" + "="*80)
    print("DOWNLOADING PUBMED (MEMORY-OPTIMIZED)")
    print("="*80)
    print("Strategy: Download + Save every 10K papers")
    print("="*80 + "\n")
    
    stream = load_dataset(
        "armanc/scientific_papers",
        "pubmed",
        split="train",
        streaming=True,
        download_config=DownloadConfig(resume_download=True)
    )
    
    BATCH_SIZE = 10000
    batch_data = []
    batch_num = 0
    total_count = 0
    
    print(f"Processing in batches of {BATCH_SIZE:,} papers...\n")
    
    for item in stream:
        batch_data.append(item)
        total_count += 1
        
        if total_count % 1000 == 0:
            print(f"  Downloaded: {total_count:,} papers...", end='\r')
        
        if len(batch_data) >= BATCH_SIZE:
            print(f"\n  Saving batch {batch_num}...")
            batch_dataset = Dataset.from_dict({
                key: [item[key] for item in batch_data]
                for key in batch_data[0].keys()
            })
            batch_dataset = batch_dataset.map(
                lambda x: {**x, 'domain': 'pubmed'},
                desc=f"Labeling batch {batch_num}"
            )
            batch_path = f"./pubmed_batch_{batch_num:03d}"
            batch_dataset.save_to_disk(batch_path)
            print(f"  ✓ Batch {batch_num} saved")
            
            batch_data = []
            batch_num += 1
    
    if batch_data:
        print(f"\n  Saving final batch {batch_num}...")
        batch_dataset = Dataset.from_dict({
            key: [item[key] for item in batch_data]
            for key in batch_data[0].keys()
        })
        batch_dataset = batch_dataset.map(
            lambda x: {**x, 'domain': 'pubmed'},
            desc=f"Labeling batch {batch_num}"
        )
        batch_path = f"./pubmed_batch_{batch_num:03d}"
        batch_dataset.save_to_disk(batch_path)
    
    print(f"\n✓ Downloaded {total_count:,} papers in {batch_num + 1} batches")
    
    # Combine batches
    print("\nCombining all batches...")
    batch_datasets = []
    for i in range(batch_num + 1):
        batch_path = f"./pubmed_batch_{i:03d}"
        if os.path.exists(batch_path):
            batch_datasets.append(load_from_disk(batch_path))
    
    combined = concatenate_datasets(batch_datasets)
    print(f"✓ Combined: {len(combined):,} papers")
    
    combined.save_to_disk("./pubmed_dataset")
    
    # Clean up
    for i in range(batch_num + 1):
        import shutil
        batch_path = f"./pubmed_batch_{i:03d}"
        if os.path.exists(batch_path):
            shutil.rmtree(batch_path)
    
    print("\n✓ PubMed saved to: ./pubmed_dataset")
    print("\nNext: Run: python prepare_datasets.py legal")


def download_legal_streaming_batched():
    """Download Legal dataset using streaming with batched saving."""
    print("\n" + "="*80)
    print("DOWNLOADING LEGAL DATASET (MEMORY-OPTIMIZED)")
    print("="*80)
    print("Strategy: Download + Save every 5K documents")
    print("Note: Using 50K subset for balanced dataset")
    print("="*80 + "\n")
    
    # Try multiple legal sources
    legal_sources = [
        ("pile-of-law/pile-of-law", "r_legaladvice", True),
        ("jonathanli/pile-of-law-sample", None, False),
    ]
    
    stream = None
    for source_name, config, needs_trust in legal_sources:
        try:
            print(f"Trying: {source_name}...")
            if config:
                stream = load_dataset(
                    source_name,
                    config,
                    split="train",
                    streaming=True,
                    trust_remote_code=needs_trust,
                    download_config=DownloadConfig(resume_download=True)
                )
            else:
                stream = load_dataset(
                    source_name,
                    split="train",
                    streaming=True,
                    download_config=DownloadConfig(resume_download=True)
                )
            print(f"✓ Connected to {source_name}")
            break
        except Exception as e:
            print(f"✗ {source_name} failed: {e}")
            continue
    
    if stream is None:
        print("\n✗ Could not connect to any legal dataset source")
        print("Continuing without legal dataset.")
        print("You can combine ArXiv + PubMed with: python download_memory_optimized.py combine")
        return
    
    BATCH_SIZE = 10000
    MAX_DOCS = 200000
    batch_data = []
    batch_num = 0
    total_count = 0
    
    print(f"\nProcessing in batches of {BATCH_SIZE:,} documents...")
    print(f"(Will stop at {MAX_DOCS:,} documents)\n")
    
    for item in stream:
        # Process to match ArXiv/PubMed format
        if 'article' in item:
            processed_item = {**item, 'domain': 'legal'}
        elif 'text' in item:
            processed_item = {'article': item['text'], 'domain': 'legal'}
        else:
            # Find any text field
            text_field = next((k for k in item.keys() 
                             if isinstance(item[k], str) and len(item[k]) > 100), None)
            if text_field:
                processed_item = {'article': item[text_field], 'domain': 'legal'}
            else:
                continue  # Skip if no text found
        
        batch_data.append(processed_item)
        total_count += 1
        
        if total_count % 500 == 0:
            print(f"  Downloaded: {total_count:,} documents...", end='\r')
        
        # Stop at max
        if total_count >= MAX_DOCS:
            print(f"\n✓ Reached {MAX_DOCS:,} documents limit")
            break
        
        # Save batch when full
        if len(batch_data) >= BATCH_SIZE:
            print(f"\n  Saving batch {batch_num}...")
            batch_dataset = Dataset.from_dict({
                key: [item[key] for item in batch_data]
                for key in batch_data[0].keys()
            })
            
            batch_path = f"./legal_batch_{batch_num:03d}"
            batch_dataset.save_to_disk(batch_path)
            print(f"  ✓ Batch {batch_num} saved")
            
            batch_data = []
            batch_num += 1
    
    # Save remaining data
    if batch_data:
        print(f"\n  Saving final batch {batch_num}...")
        batch_dataset = Dataset.from_dict({
            key: [item[key] for item in batch_data]
            for key in batch_data[0].keys()
        })
        batch_path = f"./legal_batch_{batch_num:03d}"
        batch_dataset.save_to_disk(batch_path)
    
    print(f"\n✓ Downloaded {total_count:,} documents in {batch_num + 1} batches")
    
    # Combine batches
    print("\nCombining all batches...")
    batch_datasets = []
    for i in range(batch_num + 1):
        batch_path = f"./legal_batch_{i:03d}"
        if os.path.exists(batch_path):
            batch_datasets.append(load_from_disk(batch_path))
    
    combined = concatenate_datasets(batch_datasets)
    print(f"✓ Combined: {len(combined):,} documents")
    
    combined.save_to_disk("./legal_dataset")
    
    # Clean up
    for i in range(batch_num + 1):
        import shutil
        batch_path = f"./legal_batch_{i:03d}"
        if os.path.exists(batch_path):
            shutil.rmtree(batch_path)
    
    print("\n✓ Legal dataset saved to: ./legal_dataset")
    print("\nNext: Run: python prepare_datasets.py openwebmath")


def download_openwebmath_streaming_batched():
    """Download OpenWebMath dataset using streaming with batched saving."""
    print("\n" + "="*80)
    print("DOWNLOADING OPENWEBMATH (MEMORY-OPTIMIZED)")
    print("="*80)
    print("Strategy: Download + Save every 10K documents")
    print("Note: Using 100K subset for balanced dataset")
    print("OpenWebMath: High-quality mathematical text dataset")
    print("="*80 + "\n")
    
    try:
        print("Connecting to OpenWebMath dataset...")
        stream = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
            download_config=DownloadConfig(resume_download=True)
        )
        print("✓ Connected to OpenWebMath")
    except Exception as e:
        print(f"\n✗ Could not connect to OpenWebMath: {e}")
        print("Continuing without OpenWebMath dataset.")
        return
    
    BATCH_SIZE = 10000
    MAX_DOCS = 200000
    batch_data = []
    batch_num = 0
    total_count = 0
    
    print(f"\nProcessing in batches of {BATCH_SIZE:,} documents...")
    print(f"(Will stop at {MAX_DOCS:,} documents)\n")
    
    for item in stream:
        # OpenWebMath has 'text' field, convert to 'article'
        if 'text' in item:
            processed_item = {'article': item['text'], 'domain': 'openwebmath'}
        elif 'article' in item:
            processed_item = {**item, 'domain': 'openwebmath'}
        else:
            continue
        
        batch_data.append(processed_item)
        total_count += 1
        
        if total_count % 1000 == 0:
            print(f"  Downloaded: {total_count:,} documents...", end='\r')
        
        if total_count >= MAX_DOCS:
            print(f"\n✓ Reached {MAX_DOCS:,} documents limit")
            break
        
        if len(batch_data) >= BATCH_SIZE:
            print(f"\n  Saving batch {batch_num}...")
            batch_dataset = Dataset.from_dict({
                key: [item[key] for item in batch_data]
                for key in batch_data[0].keys()
            })
            batch_path = f"./openwebmath_batch_{batch_num:03d}"
            batch_dataset.save_to_disk(batch_path)
            print(f"  ✓ Batch {batch_num} saved")
            batch_data = []
            batch_num += 1
    
    if batch_data:
        print(f"\n  Saving final batch {batch_num}...")
        batch_dataset = Dataset.from_dict({
            key: [item[key] for item in batch_data]
            for key in batch_data[0].keys()
        })
        batch_path = f"./openwebmath_batch_{batch_num:03d}"
        batch_dataset.save_to_disk(batch_path)
    
    print(f"\n✓ Downloaded {total_count:,} documents in {batch_num + 1} batches")
    
    print("\nCombining all batches...")
    batch_datasets = []
    for i in range(batch_num + 1):
        batch_path = f"./openwebmath_batch_{i:03d}"
        if os.path.exists(batch_path):
            batch_datasets.append(load_from_disk(batch_path))
    
    combined = concatenate_datasets(batch_datasets)
    combined.save_to_disk("./openwebmath_dataset")
    
    for i in range(batch_num + 1):
        import shutil
        batch_path = f"./openwebmath_batch_{i:03d}"
        if os.path.exists(batch_path):
            shutil.rmtree(batch_path)
    
    print("\n✓ OpenWebMath saved to: ./openwebmath_dataset")
    print("\nNext: python prepare_datasets.py combine")


def combine_datasets():
    """Combine ArXiv + PubMed datasets."""
    print("\n" + "="*80)
    print("COMBINING DATASETS")
    print("="*80)
    
    datasets_to_combine = []
    
    if os.path.exists("./arxiv_dataset"):
        print("Loading ArXiv...")
        arxiv = load_from_disk("./arxiv_dataset")
        datasets_to_combine.append(arxiv)
        print(f"✓ ArXiv: {len(arxiv):,}")
    else:
        print("✗ ArXiv not found")
        return
    
    if os.path.exists("./pubmed_dataset"):
        print("Loading PubMed...")
        pubmed = load_from_disk("./pubmed_dataset")
        datasets_to_combine.append(pubmed)
        print(f"✓ PubMed: {len(pubmed):,}")
    else:
        print("✗ PubMed not found")
        return
    
    if os.path.exists("./legal_dataset"):
        print("Loading legal...")
        legal = load_from_disk("./legal_dataset")
        datasets_to_combine.append(legal)
        print(f"✓ legal: {len(legal):,}")
    else:
        print("✗ legal not found")
        return
    
    if os.path.exists("./openwebmath_dataset"):
        print("Loading openwebmath...")
        openwebmath = load_from_disk("./openwebmath_dataset")
        datasets_to_combine.append(openwebmath)
        print(f"✓ openwebmath: {len(openwebmath):,}")
    else:
        print("✗ openwebmath not found")
        return

    print("\nCombining...")
    combined = concatenate_datasets(datasets_to_combine)
    
    print("Saving combined dataset...")
    combined.save_to_disk("./combined_dataset")
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"Total: {len(combined):,} papers")
    print("Saved to: ./combined_dataset")
    print("\nRun: python preprocess_v2.py --num-papers 1000")
    print("="*80)


def show_usage():
    print("\n" + "="*80)
    print("MEMORY-OPTIMIZED DOWNLOADER")
    print("="*80)
    print("\nUsage:")
    print("  python download_memory_optimized.py arxiv")
    print("  python download_memory_optimized.py pubmed")
    print("  python download_memory_optimized.py combine")
    print("\nSaves in batches of 10K papers to prevent memory issues")
    print("="*80 + "\n")


def main():
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    try:
        if command == "arxiv":
            download_arxiv_streaming_batched()
        elif command == "pubmed":
            download_pubmed_streaming_batched()
        elif command == "legal":
            download_legal_streaming_batched()
        elif command == "openwebmath":
            download_openwebmath_streaming_batched()
        elif command == "combine":
            combine_datasets()
        else:
            print(f"Unknown command: {command}")
            show_usage()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()