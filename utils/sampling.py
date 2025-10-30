#!/usr/bin/env python3
"""
create_balanced_sample.py - Customizable dataset sampler for all 4 domains

Creates balanced samples with custom counts per domain.

Features:
- Specify custom count for each domain (or use defaults)
- Automatically clamps to available dataset size
- Shows what's available vs what's requested
- Reproducible with fixed random seed

Usage (defaults: 2,500 per domain = 10K total):
    python create_balanced_sample.py

Usage (custom counts):
    python create_balanced_sample.py --arxiv 1000 --pubmed 1000 --legal 1000 --openwebmath 1000
    python create_balanced_sample.py --arxiv 5000 --pubmed 5000 --legal 5000 --openwebmath 5000
    python create_balanced_sample.py --only arxiv,legal --arxiv 3000 --legal 3000

Examples:
    # Small quick test (4K total)
    python create_balanced_sample.py --arxiv 1000 --pubmed 1000 --legal 1000 --openwebmath 1000
    
    # Medium test (10K total)
    python create_balanced_sample.py  # Default
    
    # Large test (20K total)
    python create_balanced_sample.py --arxiv 5000 --pubmed 5000 --legal 5000 --openwebmath 5000
    
    # Only specific domains
    python create_balanced_sample.py --only arxiv,pubmed --arxiv 5000 --pubmed 5000
"""

import argparse
import logging
from pathlib import Path
import random

from datasets import load_from_disk, concatenate_datasets

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_custom_sample(
    sample_counts: dict,
    datasets_dir: str = "raw_datasets",
    output_dir: str = "combined_dataset_sample",
    only_domains: list = None
):
    """
    Create custom sample from specified domains with specified counts.
    
    Args:
        sample_counts: Dict like {'arxiv': 2500, 'pubmed': 2500, ...}
        datasets_dir: Path to raw downloaded datasets
        output_dir: Where to save combined sample
        only_domains: List of domains to use (None = use all available)
    """
    
    logger.info("\n" + "="*80)
    logger.info("CREATING CUSTOM SAMPLE")
    logger.info("="*80)
    
    datasets_path = Path(datasets_dir)
    output_path = Path(output_dir)
    
    all_samples = []
    total_requested = 0
    total_sampled = 0
    
    domains_info = {
        'arxiv': 'ArXiv (Scientific Papers)',
        'pubmed': 'PubMed (Biomedical Papers)',
        'legal': 'Legal (Law Documents)',
        'openwebmath': 'OpenWebMath (Mathematical Text)'
    }
    
    # Determine which domains to process
    domains_to_process = only_domains if only_domains else list(domains_info.keys())
    
    logger.info(f"\nRequested domains: {', '.join(domains_to_process)}")
    logger.info("\n[Domain Sampling Plan]:")
    
    # First pass: show what's available vs requested
    domain_status = {}
    for domain_name in domains_to_process:
        domain_path = datasets_path / domain_name
        
        if not domain_path.exists():
            logger.warning(f"  ⚠ {domain_name}: NOT FOUND (skipping)")
            domain_status[domain_name] = {'available': 0, 'requested': 0, 'will_sample': 0}
            continue
        
        try:
            dataset = load_from_disk(str(domain_path))
            available = len(dataset)
            requested = sample_counts.get(domain_name, 0)
            will_sample = min(available, requested)
            
            domain_status[domain_name] = {
                'available': available,
                'requested': requested,
                'will_sample': will_sample
            }
            
            status_str = f"  {domain_name.upper()}"
            logger.info(f"{status_str}:")
            logger.info(f"    Available: {available:,}")
            logger.info(f"    Requested: {requested:,}")
            if will_sample < requested:
                logger.info(f"    Will sample: {will_sample:,} (capped to available)")
            else:
                logger.info(f"    Will sample: {will_sample:,}")
            
            total_requested += requested
            
        except Exception as e:
            logger.error(f"  ✗ Error loading {domain_name}: {e}")
            domain_status[domain_name] = {'available': 0, 'requested': 0, 'will_sample': 0}
    
    logger.info(f"\nTotal requested: {total_requested:,} papers")
    
    # Second pass: actually sample
    logger.info(f"\n[Sampling]:")
    
    for domain_name in domains_to_process:
        domain_path = datasets_path / domain_name
        status = domain_status.get(domain_name, {})
        
        if status['will_sample'] == 0:
            continue
        
        try:
            dataset = load_from_disk(str(domain_path))
            
            # Random indices
            indices = random.sample(range(len(dataset)), status['will_sample'])
            sample = dataset.select(indices)
            
            all_samples.append(sample)
            total_sampled += len(sample)
            
            logger.info(f"  ✓ {domain_name}: sampled {len(sample):,} papers")
        
        except Exception as e:
            logger.error(f"  ✗ Error sampling {domain_name}: {e}")
            continue
    
    if not all_samples:
        logger.error("\n✗ No datasets to sample!")
        return
    
    # Combine all samples
    logger.info(f"\n[Combining]:")
    logger.info(f"  Combining {len(all_samples)} domain samples...")
    combined = concatenate_datasets(all_samples)
    
    logger.info(f"  Total papers: {len(combined):,}")
    
    # Verify domain distribution
    logger.info(f"\n[Final Domain Distribution]:")
    domains_found = combined.unique('domain')
    for domain in sorted(domains_found):
        count = len(combined.filter(lambda x: x['domain'] == domain))
        pct = 100 * count / len(combined)
        logger.info(f"  {domain}: {count:,} ({pct:.1f}%)")
    
    # Save
    logger.info(f"\n[Saving]:")
    logger.info(f"  Saving to {output_path}...")
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    combined.save_to_disk(str(output_path))
    
    logger.info(f"  ✓ Saved {len(combined):,} papers")
    logger.info(f"\n✓ Sample creation complete!")
    logger.info(f"  Location: {output_path}")
    
    logger.info(f"\nNext step:")
    logger.info(f"  python preprocess_updated.py --show-examples")
    
    logger.info("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Create custom balanced dataset sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 2,500 per domain (10K total)
  python create_balanced_sample.py
  
  # Custom: 1K per domain (4K total)
  python create_balanced_sample.py --arxiv 1000 --pubmed 1000 --legal 1000 --openwebmath 1000
  
  # Large: 5K per domain (20K total)
  python create_balanced_sample.py --arxiv 5000 --pubmed 5000 --legal 5000 --openwebmath 5000
  
  # Only specific domains
  python create_balanced_sample.py --only arxiv,legal --arxiv 3000 --legal 3000
        """
    )
    
    parser.add_argument(
        '--arxiv',
        type=int,
        default=2500,
        help='Number of ArXiv papers to sample (default: 2500)'
    )
    parser.add_argument(
        '--pubmed',
        type=int,
        default=2500,
        help='Number of PubMed papers to sample (default: 2500)'
    )
    parser.add_argument(
        '--legal',
        type=int,
        default=2500,
        help='Number of Legal documents to sample (default: 2500)'
    )
    parser.add_argument(
        '--openwebmath',
        type=int,
        default=2500,
        help='Number of OpenWebMath documents to sample (default: 2500)'
    )
    parser.add_argument(
        '--only',
        type=str,
        default=None,
        help='Comma-separated list of domains to use (e.g., "arxiv,legal")'
    )
    parser.add_argument(
        '--datasets-dir',
        default='raw_datasets',
        help='Path to raw datasets directory (default: raw_datasets)'
    )
    parser.add_argument(
        '--output-dir',
        default='combined_dataset_sample',
        help='Output directory for combined sample (default: combined_dataset_sample)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Build sample counts dict
    sample_counts = {
        'arxiv': args.arxiv,
        'pubmed': args.pubmed,
        'legal': args.legal,
        'openwebmath': args.openwebmath
    }
    
    # Parse only_domains if specified
    only_domains = None
    if args.only:
        only_domains = [d.strip().lower() for d in args.only.split(',')]
        logger.info(f"Using only domains: {only_domains}")
    
    create_custom_sample(
        sample_counts=sample_counts,
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        only_domains=only_domains
    )


if __name__ == "__main__":
    main()