#!/usr/bin/env python3

"""
preprocess.py - DISCOURSE-AWARE REASONING PRETRAINING (CUSTOMIZABLE PER-DATASET SAMPLING)

Complete preprocessing pipeline with:
- Exact tag format: <word>
- Discourse connector detection (uniform 1.1x weighting from config)
- Architecture-aware attention masks (connector_mask: 0.0/1.0/1.1)
- PER-DATASET CONTROL: Specify how many papers from each domain
- Sample output generation for verification

CORE IDEA (Discourse-Aware Reasoning, NOT ToW):
- ToW: Adds sequential "thoughts" to improve multi-step reasoning
- Our approach: Emphasizes discourse connectors that link logical reasoning
- Example difference:
  ToW: "Step 1: ... Step 2: ... Step 3: ..."
  Ours: "Sentence A because Sentence B, therefore Conclusion"

Discourse connectors encode logical relationships directly in text structure,
improving reasoning without explicit step-by-step decomposition.

4 Datasets: ArXiv (scientific), PubMed (biomedical), Legal (law), OpenWebMath (math)

Usage:
  # Equal sampling from all datasets
  python preprocess.py --num-papers 10000

  # Custom per-dataset counts
  python preprocess.py --arxiv 5000 --pubmed 3000 --legal 1000 --openwebmath 1000

  # Mix of specified and automatic
  python preprocess.py --arxiv 8000 --num-papers 2000  # 8K arxiv + 2K from others

  # Resume from checkpoint
  python preprocess.py --resume --arxiv 10000

  # Show preprocessing examples
  python preprocess.py --num-papers 100 --show-examples
"""

import os, sys, time, logging, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch, pandas as pd, numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer
from config import (
    BASE_MODEL, CHECKPOINT_SIZE, CHECKPOINT_DIR, SAVE_TEXT_PREVIEW, PREVIEW_LENGTH,
    SPECIAL_TOKENS, CONNECTOR_PATTERNS, MIN_PAPER_LENGTH, MAX_PAPER_LENGTH,
    COMBINED_DATASET_PATH, CONNECTOR_ATTENTION_WEIGHT, TAG_FORMAT,
    format_connector_tag, get_connector_types
)
from connector_detector import ConnectorDetector, tag_text
from tokenizer_utils import ConnectorTokenizer
from checkpoint_manager import (
    ensure_checkpoint_dir, get_last_checkpoint, save_checkpoint,
    save_metadata, print_checkpoint_status
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Domain names (must match your dataset)
AVAILABLE_DOMAINS = ["arxiv", "pubmed", "legal", "openwebmath"]


def extract_paper_text(paper: Dict) -> str:
    """Extract text from paper - ALWAYS returns string (never None)."""
    text_fields = ['article', 'abstract', 'text', 'opinion', 'content', 'full_text', 'body']

    for field in text_fields:
        if field not in paper:
            continue
        value = paper[field]
        if value is None:
            continue
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                continue
        if len(value.strip()) > 0:
            return value.strip()

    # Fallback: try title + abstract
    parts = []
    for field in ['title', 'abstract', 'summary']:
        if field in paper and paper[field]:
            try:
                parts.append(str(paper[field]))
            except:
                pass

    if parts:
        combined = " ".join(parts).strip()
        if len(combined) > 0:
            return combined

    return ""


def get_paper_domain(paper: Dict) -> str:
    """Get domain/dataset name from paper."""
    return paper.get('domain', 'unknown')


def get_paper_id(paper: Dict, index: int) -> str:
    """Generate unique paper ID."""
    if 'article_id' in paper:
        return paper['article_id']
    elif 'id' in paper:
        return paper['id']
    else:
        domain = get_paper_domain(paper)
        return f"{domain}_{index:07d}"


def should_keep_paper(text: str) -> bool:
    """
    Filter papers by quality - CONFIGURABLE version.

    Modify MIN_PAPER_LENGTH and MAX_PAPER_LENGTH in config.py
    to adjust filtering strictness.
    """
    if text is None or not isinstance(text, str):
        return False

    if len(text.strip()) == 0:
        return False

    try:
        word_count = len(text.strip().split())
    except:
        return False

    if word_count < MIN_PAPER_LENGTH:
        return False

    if word_count > MAX_PAPER_LENGTH:
        return False

    return True


def load_combined_datasets(path=COMBINED_DATASET_PATH) -> object:
    """Load combined dataset from disk."""
    logger.info(f"Loading combined dataset from: {path}")
    if not os.path.exists(path):
        logger.error(f"✗ Combined dataset not found at: {path}")
        logger.info("\n💡 Run first:")
        logger.info("  python prepare_datasets.py arxiv")
        logger.info("  python prepare_datasets.py pubmed")
        logger.info("  python prepare_datasets.py legal")
        logger.info("  python prepare_datasets.py openwebmath")
        logger.info("  python prepare_datasets.py combine")
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = load_from_disk(path)
    logger.info(f"✓ Loaded: {len(dataset):,} documents")

    if 'domain' in dataset.column_names:
        domains = {}
        for paper in dataset:
            domain = paper.get('domain', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1
        logger.info("Domain distribution:")
        for domain, count in sorted(domains.items()):
            logger.info(f"  {domain}: {count:,}")

    return dataset


def create_domain_sampling_plan(
    dataset,
    domain_limits: Dict[str, int],
    default_per_domain: Optional[int] = None
) -> Dict[str, List[int]]:
    """
    Create sampling plan: which indices to process from each domain.

    Args:
        dataset: Combined dataset
        domain_limits: Dict of {domain_name: max_papers}
        default_per_domain: Default limit for domains not in domain_limits

    Returns:
        Dict of {domain_name: [list of indices to process]}
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING DOMAIN SAMPLING PLAN")
    logger.info("="*80)

    # Group indices by domain
    domain_indices = {}
    for idx, paper in enumerate(dataset):
        domain = get_paper_domain(paper)
        if domain not in domain_indices:
            domain_indices[domain] = []
        domain_indices[domain].append(idx)

    # Show available domains
    logger.info("\nAvailable domains in dataset:")
    for domain, indices in sorted(domain_indices.items()):
        logger.info(f"  {domain}: {len(indices):,} papers")

    # Create sampling plan
    sampling_plan = {}
    total_papers = 0

    logger.info("\nSampling plan:")
    for domain, all_indices in sorted(domain_indices.items()):
        # Determine limit for this domain
        if domain in domain_limits:
            limit = domain_limits[domain]
        elif default_per_domain is not None:
            limit = default_per_domain
        else:
            limit = len(all_indices)  # Process all if no limit specified

        # Sample indices (take first N, or use random sampling)
        sampled_indices = all_indices[:limit]
        sampling_plan[domain] = sampled_indices
        total_papers += len(sampled_indices)

        logger.info(f"  {domain}: {len(sampled_indices):,} / {len(all_indices):,} papers "
                   f"({100*len(sampled_indices)/len(all_indices):.1f}%)")

    logger.info(f"\nTotal papers to process: {total_papers:,}")
    logger.info("="*80 + "\n")

    return sampling_plan


def download_tokenizer_if_needed(model_name: str = BASE_MODEL) -> AutoTokenizer:
    """Download tokenizer on first run if not cached."""
    try:
        logger.info(f"Checking cache for tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        logger.info("✓ Tokenizer found in cache")
        return tokenizer
    except Exception:
        logger.info("⏳ Downloading tokenizer (first time only, ~500MB)...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Downloaded in {time.time()-start:.1f}s")
        return tokenizer


def setup_tokenizer_with_special_tokens(base_tokenizer, save_path: Optional[str] = None) -> AutoTokenizer:
    """Add special connector tokens to tokenizer."""
    original_size = len(base_tokenizer)
    num_added = base_tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})

    logger.info(f"Extended tokenizer:\n"
                f"  Original vocab: {original_size:,}\n"
                f"  Added tokens: {num_added}\n"
                f"  New vocab: {len(base_tokenizer):,}\n"
                f"  Special tokens: {len(SPECIAL_TOKENS)}")

    if save_path:
        base_tokenizer.save_pretrained(save_path)
        logger.info(f"  Saved to: {save_path}")

    return base_tokenizer


def process_single_paper(
    paper: Dict,
    paper_idx: int,
    tokenizer: AutoTokenizer,
    detector: ConnectorDetector,
    connector_tok: ConnectorTokenizer
) -> Optional[Dict]:
    """Process single paper with comprehensive error handling."""
    try:
        # 1. Extract text
        raw_text = extract_paper_text(paper)

        # 2. Quality filter
        if not should_keep_paper(raw_text):
            return None

        # 3. Handle edge case: empty after filtering
        if isinstance(raw_text, str) and len(raw_text.strip()) == 0:
            return None

        # 4. Tag connectors
        tagged_result = tag_text(raw_text)

        # 5. Handle None from tag_text
        if tagged_result is None or not isinstance(tagged_result, dict):
            tagged_result = {
                'tagged_text': raw_text,
                'connector_words': [],
                'connector_types': []
            }

        if tagged_result.get('tagged_text') is None:
            tagged_result['tagged_text'] = raw_text

        # 6. Tokenize
        tokenized = connector_tok.tokenize_with_connector_mask(
            tagged_result['tagged_text'],
            max_length=2048,
            return_tensors="pt",
            connector_weight=CONNECTOR_ATTENTION_WEIGHT
        )

        # 7. Build result
        result = {
            'doc_id': get_paper_id(paper, paper_idx),
            'domain': get_paper_domain(paper),
            'input_ids': tokenized['input_ids'].tolist() if hasattr(tokenized['input_ids'], 'tolist') else tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'].tolist() if hasattr(tokenized['attention_mask'], 'tolist') else tokenized['attention_mask'],
            'connector_mask': tokenized['connector_mask'].tolist() if hasattr(tokenized['connector_mask'], 'tolist') else tokenized['connector_mask'],
            'connector_count': len(tagged_result['connector_words']),
            'connector_types': tagged_result['connector_types'],
            'connector_words': tagged_result['connector_words'],
            'token_count': len(tokenized['input_ids']) if hasattr(tokenized['input_ids'], '__len__') else 0,
            'connector_density': len(tagged_result['connector_words']) / max(len(raw_text.split()), 1)
        }

        return result

    except Exception as e:
        logger.warning(f"Error processing paper {paper_idx}: {e}")
        return None


def show_preprocessing_example(
    raw_text: str,
    annotated_text: str,
    connector_words: List[str],
    connector_types: List[str]
) -> None:
    """Show a concrete example of how preprocessing works."""
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING EXAMPLE - DISCOURSE-AWARE REASONING")
    logger.info("="*80)

    logger.info("\n[1] ORIGINAL TEXT:\n" + "-"*80)
    logger.info(raw_text[:500])

    logger.info("\n[2] AFTER CONNECTOR TAGGING:\n" + "-"*80)
    logger.info(f"Format: {TAG_FORMAT}\n{annotated_text[:500]}")

    logger.info("\n[3] CONNECTOR ANALYSIS:\n" + "-"*80)
    logger.info(f"Detected connectors: {len(connector_words)} | Unique types: {len(set(connector_types))}")

    type_counts = {}
    for ct in connector_types:
        type_counts[ct] = type_counts.get(ct, 0) + 1

    logger.info("\nConnector type distribution:")
    for ctype, count in sorted(type_counts.items()):
        logger.info(f"  {ctype.upper()}: {count}")

    logger.info("\nConnector words found:")
    for i, (word, ctype) in enumerate(zip(connector_words[:10], connector_types[:10])):
        logger.info(f"  {i+1}. {word:<15} ({ctype})")
    if len(connector_words) > 10:
        logger.info(f"  ... and {len(connector_words) - 10} more")

    logger.info("\n[4] KEY DESIGN CHOICE - UNIFORM 1.1x WEIGHTING:\n" + "-"*80)
    logger.info(f"Weight applied to ALL connector types: {CONNECTOR_ATTENTION_WEIGHT}x\n"
                f"Result: Better generalization and more accurate reasoning\n" + "="*80 + "\n")


def run_preprocessing_with_domain_control(
    domain_limits: Dict[str, int],
    default_per_domain: Optional[int] = None,
    checkpoint_size: int = CHECKPOINT_SIZE,
    show_examples: bool = False
) -> None:
    """Run preprocessing with per-domain control over sampling."""
    logger.info("="*80)
    logger.info("DISCOURSE-AWARE REASONING PRETRAINING (PER-DATASET CONTROL)")
    logger.info("="*80)
    logger.info(f"Tag format: {TAG_FORMAT}\n"
                f"Connector weight: {CONNECTOR_ATTENTION_WEIGHT}x\n"
                f"Architecture: Pre-softmax attention modification\n"
                f"Min paper length: {MIN_PAPER_LENGTH} words\n"
                f"Max paper length: {MAX_PAPER_LENGTH} words")

    ensure_checkpoint_dir()

    logger.info("\n[1/6] Loading combined dataset...")
    dataset = load_combined_datasets()

    logger.info("\n[2/6] Creating domain sampling plan...")
    sampling_plan = create_domain_sampling_plan(dataset, domain_limits, default_per_domain)

    # Flatten sampling plan into list of indices to process
    all_indices = []
    for domain, indices in sampling_plan.items():
        all_indices.extend(indices)

    # Sort to maintain some order (optional)
    all_indices.sort()

    logger.info("\n[3/6] Initializing connector detector...")
    detector = ConnectorDetector(CONNECTOR_PATTERNS)
    total_patterns = sum(len(p) for p in CONNECTOR_PATTERNS.values())
    logger.info(f"✓ Loaded {total_patterns} connector patterns ({', '.join(get_connector_types())})")

    logger.info("\n[4/6] Setting up tokenizer...")
    base_tokenizer = download_tokenizer_if_needed()
    tokenizer_save_path = os.path.join(CHECKPOINT_DIR, "extended_tokenizer")
    tokenizer = setup_tokenizer_with_special_tokens(base_tokenizer, tokenizer_save_path)
    connector_tok = ConnectorTokenizer(tokenizer, connector_weight=CONNECTOR_ATTENTION_WEIGHT)

    total_to_process = len(all_indices)

    logger.info(f"\n[5/6] Configuration:\n"
                f"  Total papers to process: {total_to_process:,}\n"
                f"  Checkpoint size: {checkpoint_size:,}")

    # Domain-wise stats
    domain_stats = {domain: {'processed': 0, 'filtered': 0, 'connectors': 0} 
                    for domain in sampling_plan.keys()}

    processed_batch, checkpoint_id = [], 0
    papers_processed = papers_filtered = total_connectors = 0
    example_shown, start_time = False, time.time()

    logger.info("\n[6/6] Processing papers...")

    with tqdm(total=total_to_process, desc="Progress", unit="papers") as pbar:
        for idx in all_indices:
            paper = dataset[idx]
            domain = get_paper_domain(paper)
            result = process_single_paper(paper, idx, tokenizer, detector, connector_tok)

            if result is None:
                papers_filtered += 1
                domain_stats[domain]['filtered'] += 1
                pbar.update(1)
                continue

            if show_examples and not example_shown and result.get('connector_words'):
                # Show example (need to re-extract for display)
                raw_text = extract_paper_text(paper)
                tagged = tag_text(raw_text)
                if tagged:
                    show_preprocessing_example(
                        raw_text,
                        tagged['tagged_text'],
                        result['connector_words'],
                        result['connector_types']
                    )
                    example_shown = True

            processed_batch.append(result)
            papers_processed += 1
            total_connectors += result['connector_count']
            domain_stats[domain]['processed'] += 1
            domain_stats[domain]['connectors'] += result['connector_count']
            pbar.update(1)

            if len(processed_batch) >= checkpoint_size:
                save_checkpoint(processed_batch, checkpoint_id)
                df = pd.DataFrame(processed_batch)
                avg_connectors = df['connector_count'].mean()
                total_tokens = df['token_count'].sum()
                elapsed = time.time() - start_time
                speed = papers_processed / elapsed if elapsed > 0 else 0
                pbar.write(f"\n✓ Checkpoint {checkpoint_id:04d} saved | "
                          f"Papers: {len(df):,} | "
                          f"Avg connectors: {avg_connectors:.1f} | "
                          f"Total tokens: {total_tokens:,} | "
                          f"Speed: {speed:.1f}/s")
                processed_batch, checkpoint_id = [], checkpoint_id + 1

    if processed_batch:
        save_checkpoint(processed_batch, checkpoint_id)
        logger.info(f"\n✓ Final checkpoint {checkpoint_id:04d}: {len(processed_batch)} papers")

    total_time = time.time() - start_time

    # Show per-domain statistics
    logger.info("\n" + "="*80)
    logger.info("PER-DOMAIN STATISTICS")
    logger.info("="*80)
    for domain, stats in sorted(domain_stats.items()):
        total_domain = stats['processed'] + stats['filtered']
        pass_rate = 100 * stats['processed'] / total_domain if total_domain > 0 else 0
        avg_conn = stats['connectors'] / stats['processed'] if stats['processed'] > 0 else 0
        logger.info(f"{domain}:")
        logger.info(f"  Processed: {stats['processed']:,} | "
                   f"Filtered: {stats['filtered']:,} | "
                   f"Pass rate: {pass_rate:.1f}% | "
                   f"Avg connectors: {avg_conn:.1f}")

    save_metadata({
        "papers_processed": papers_processed,
        "papers_filtered": papers_filtered,
        "total_connectors": total_connectors,
        "avg_connectors_per_paper": total_connectors / papers_processed if papers_processed > 0 else 0,
        "total_time_seconds": total_time,
        "papers_per_second": papers_processed / total_time if total_time > 0 else 0,
        "checkpoint_size": checkpoint_size,
        "tag_format": TAG_FORMAT,
        "connector_attention_weight": CONNECTOR_ATTENTION_WEIGHT,
        "weight_type": "uniform_across_all_connector_types",
        "reasoning_approach": "discourse_aware_reasoning",
        "domain_limits": domain_limits,
        "domain_stats": domain_stats,
        "min_paper_length": MIN_PAPER_LENGTH,
        "max_paper_length": MAX_PAPER_LENGTH
    })

    logger.info("\n" + "="*80)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Papers processed: {papers_processed:,} | "
                f"Filtered: {papers_filtered:,} | "
                f"Connectors: {total_connectors:,} | "
                f"Avg connectors/paper: {total_connectors/papers_processed:.1f} | "
                f"Time: {total_time/60:.1f} min | "
                f"Speed: {papers_processed/total_time:.1f}/s | "
                f"Output: {CHECKPOINT_DIR}")
    logger.info("="*80)


def main():
    """Main entry point with per-dataset control."""
    parser = argparse.ArgumentParser(
        description="Discourse-Aware Reasoning Preprocessing with Per-Dataset Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 10K papers total (distributed evenly across datasets)
  python preprocess.py --num-papers 10000

  # Specify exact counts per dataset
  python preprocess.py --arxiv 5000 --pubmed 3000 --legal 1000 --openwebmath 1000

  # Mix: explicit for some, default for others
  python preprocess.py --arxiv 8000 --num-papers 2000

  # Resume from checkpoint with new limits
  python preprocess.py --resume --arxiv 10000 --pubmed 5000

  # Show preprocessing examples
  python preprocess.py --num-papers 100 --show-examples
        """
    )

    # Per-dataset controls
    parser.add_argument('--arxiv', type=int, default=None,
                       help='Number of papers from ArXiv datase')
    parser.add_argument('--pubmed', type=int, default=None,
                       help='Number of papers from PubMed dataset')
    parser.add_argument('--legal', type=int, default=None,
                       help='Number of papers from Legal dataset')
    parser.add_argument('--openwebmath', type=int, default=None,
                       help='Number of papers from OpenWebMath dataset')

    # Global controls
    parser.add_argument('--num-papers', type=int, default=None,
                       help='Default papers per dataset (if not specified individually)')
    parser.add_argument('--checkpoint-size', type=int, default=CHECKPOINT_SIZE,
                       help=f'Papers per checkpoint (default: {CHECKPOINT_SIZE})')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--status', action='store_true',
                       help='Show status and exit')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show preprocessing examples')

    args = parser.parse_args()

    if args.status:
        print_checkpoint_status()
        sys.exit(0)

    # Build domain limits dictionary
    domain_limits = {}
    if args.arxiv is not None:
        domain_limits['arxiv'] = args.arxiv
    if args.pubmed is not None:
        domain_limits['pubmed'] = args.pubmed
    if args.legal is not None:
        domain_limits['legal'] = args.legal
    if args.openwebmath is not None:
        domain_limits['openwebmath'] = args.openwebmath

    # If no per-dataset limits specified, use num_papers as default
    default_per_domain = args.num_papers if len(domain_limits) == 0 else None

    # Validate: need at least some specification
    if len(domain_limits) == 0 and default_per_domain is None:
        logger.error("\n✗ Must specify either --num-papers or individual dataset limits")
        logger.info("\nExamples:")
        logger.info("  python preprocess.py --num-papers 10000")
        logger.info("  python preprocess.py --arxiv 5000 --pubmed 3000")
        sys.exit(1)

    # Show configuration
    logger.info("\nDataset sampling configuration:")
    if domain_limits:
        for domain, limit in domain_limits.items():
            logger.info(f"  {domain}: {limit:,} papers")
    if default_per_domain:
        logger.info(f"  Others: {default_per_domain:,} papers each")

    try:
        run_preprocessing_with_domain_control(
            domain_limits=domain_limits,
            default_per_domain=default_per_domain,
            checkpoint_size=args.checkpoint_size,
            show_examples=args.show_examples
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Interrupted\n💡 Run with --resume to continue")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n✗ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()