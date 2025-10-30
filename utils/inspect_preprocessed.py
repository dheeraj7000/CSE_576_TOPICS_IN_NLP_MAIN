#!/usr/bin/env python3
"""
inspect_preprocessed.py - CRD UPDATED: Checkpoint Inspection & Visualization

Inspect and visualize preprocessed checkpoint files with focus on:
- Exact tag format: <connector type="X"> word </connector>
- Connector mask validation (0.0 padding, 1.0 regular, 1.1 connectors)
- Connector type distribution
- Before/after transformations

Usage:
    python inspect_preprocessed.py                    # First checkpoint
    python inspect_preprocessed.py --checkpoint 0     # Specific checkpoint
    python inspect_preprocessed.py --paper 5          # Specific paper
    python inspect_preprocessed.py --compare          # Show before/after
    python inspect_preprocessed.py --stats            # Statistics
    python inspect_preprocessed.py --mask             # Analyze masks
"""

import argparse
import logging
from typing import Dict, List, Union, Optional

import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from config import (
    CHECKPOINT_DIR,
    SPECIAL_TOKENS,
    CONNECTOR_ATTENTION_WEIGHT,
    TAG_FORMAT,
)
from checkpoint_manager import (
    load_checkpoint,
    load_all_checkpoints,
    get_checkpoint_stats,
    get_last_checkpoint,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# INSPECTION FUNCTIONS
# ============================================================================

def visualize_connector_mask(
    input_ids: List[int],
    connector_mask: Union[List[float], np.ndarray],
    tokenizer: AutoTokenizer,
    max_tokens: int = 50
) -> str:
    """
    Visualize connector mask with color coding.
    
    Shows:
    - Padding tokens (mask = 0.0): [PADDING]
    - Regular tokens (mask = 1.0): NORMAL
    - Connector tokens (mask = 1.1): **BOOSTED**
    
    Args:
        input_ids: Token IDs
        connector_mask: Mask values
        tokenizer: Tokenizer to decode tokens
        max_tokens: Max tokens to show
    
    Returns:
        Formatted string visualization
    """
    
    # Handle numpy arrays
    if isinstance(connector_mask, np.ndarray):
        connector_mask = connector_mask.tolist()
    
    visualization = []
    
    for i, (token_id, mask_val) in enumerate(zip(input_ids[:max_tokens], connector_mask[:max_tokens])):
        try:
            token_str = tokenizer.decode([token_id])
        except:
            token_str = f"[TOKEN_{token_id}]"
        
        # Classify based on mask value
        if mask_val == 0.0:
            formatted = f"[PADDING]"
        elif mask_val > 1.05:  # Connector (1.1)
            formatted = f"**{token_str}** (1.1x)"
        else:  # Regular (1.0)
            formatted = token_str
        
        visualization.append(formatted)
    
    return " ".join(visualization)


def show_paper_details(
    paper: Dict,
    tokenizer: Optional[AutoTokenizer] = None,
    show_preview: bool = True
) -> None:
    """
    Display detailed information about a single paper.
    
    Shows:
    - Metadata (ID, domain, token count)
    - Connector analysis (count, types, density)
    - Mask visualization
    - Text preview with tags
    
    Args:
        paper: Paper dict from checkpoint
        tokenizer: Optional tokenizer for decoding
        show_preview: Show text preview
    """
    
    logger.info("\n" + "="*80)
    logger.info("PAPER DETAILS")
    logger.info("="*80)
    
    # Basic info
    logger.info(f"\n[Metadata]:")
    logger.info(f"  Doc ID: {paper.get('doc_id', 'N/A')}")
    logger.info(f"  Domain: {paper.get('domain', 'N/A')}")
    logger.info(f"  Token count: {paper.get('token_count', 'N/A')}")
    
    # Connector info
    logger.info(f"\n[Connector Analysis]:")
    logger.info(f"  Total connectors: {paper.get('connector_count', 0)}")
    
    connector_types = paper.get('connector_types', [])
    if connector_types:
        logger.info(f"  Types found: {set(connector_types)}")
        type_counts = {}
        for ct in connector_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        for ct, count in sorted(type_counts.items()):
            logger.info(f"    {ct}: {count}")
    
    connector_words = paper.get('connector_words', [])
    if connector_words:
        logger.info(f"  Connector words (first 10): {connector_words[:10]}")
    
    # Mask analysis
    connector_mask = paper.get('connector_mask', [])
    if connector_mask:
        if isinstance(connector_mask, np.ndarray):
            connector_mask = connector_mask.tolist()
        
        logger.info(f"\n[Connector Mask Analysis]:")
        logger.info(f"  Mask length: {len(connector_mask)}")
        
        # Count mask values
        padding_count = sum(1 for m in connector_mask if m == 0.0)
        regular_count = sum(1 for m in connector_mask if m == 1.0)
        boosted_count = sum(1 for m in connector_mask if m > 1.05)
        
        logger.info(f"  Padding (0.0): {padding_count}")
        logger.info(f"  Regular (1.0): {regular_count}")
        logger.info(f"  Boosted ({CONNECTOR_ATTENTION_WEIGHT}x): {boosted_count}")
        
        # Unique values
        unique_vals = sorted(set(connector_mask))
        logger.info(f"  Unique values: {unique_vals}")
    
    # Text preview
    if show_preview:
        logger.info(f"\n[Text Preview]:")
        
        if 'original_text' in paper and paper['original_text']:
            logger.info(f"  Original text (first 200 chars):")
            logger.info(f"    {paper['original_text'][:200]}...")
        
        if 'annotated_text' in paper and paper['annotated_text']:
            logger.info(f"\n  Annotated text (first 200 chars):")
            logger.info(f"  Format: {TAG_FORMAT}")
            logger.info(f"    {paper['annotated_text'][:200]}...")
    
    logger.info("\n" + "="*80)


def show_checkpoint_overview(
    checkpoint_id: int = 0,
    checkpoint_dir: str = CHECKPOINT_DIR
) -> None:
    """
    Display overview of a checkpoint.
    
    Shows statistics for all papers in checkpoint.
    
    Args:
        checkpoint_id: Checkpoint number
        checkpoint_dir: Path to checkpoint directory
    """
    
    try:
        df = load_checkpoint(checkpoint_id, checkpoint_dir)
    except Exception as e:
        logger.error(f"✗ Error loading checkpoint: {e}")
        return
    
    logger.info("\n" + "="*80)
    logger.info(f"CHECKPOINT {checkpoint_id:04d} OVERVIEW")
    logger.info("="*80)
    
    logger.info(f"\n[Statistics]:")
    logger.info(f"  Papers: {len(df)}")
    logger.info(f"  Avg tokens/paper: {df['token_count'].mean():.0f}")
    logger.info(f"  Total tokens: {df['token_count'].sum():,}")
    
    # Connector stats
    if 'connector_count' in df.columns:
        logger.info(f"  Avg connectors/paper: {df['connector_count'].mean():.1f}")
        logger.info(f"  Total connectors: {df['connector_count'].sum():,}")
        
        total_tokens = df['token_count'].sum()
        total_connectors = df['connector_count'].sum()
        if total_tokens > 0:
            logger.info(f"  Connector density: {total_connectors/total_tokens*100:.2f}%")
    
    # Domain distribution
    if 'domain' in df.columns:
        logger.info(f"\n[Domain Distribution]:")
        domains = df['domain'].value_counts()
        for domain, count in domains.items():
            pct = 100 * count / len(df)
            logger.info(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Connector type distribution
    if 'connector_types' in df.columns:
        logger.info(f"\n[Connector Type Distribution]:")
        all_types = []
        for types_list in df['connector_types']:
            if isinstance(types_list, list):
                all_types.extend(types_list)
        
        if all_types:
            type_counts = {}
            for ct in all_types:
                type_counts[ct] = type_counts.get(ct, 0) + 1
            
            total = sum(type_counts.values())
            for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / total
                logger.info(f"  {ct}: {count} ({pct:.1f}%)")
    
    # Mask validation
    if 'connector_mask' in df.columns:
        logger.info(f"\n[Connector Mask Validation]:")
        logger.info(f"  ✓ All papers have connector_mask field")
        
        # Sample mask values
        sample_mask = df['connector_mask'].iloc[0]
        if isinstance(sample_mask, np.ndarray):
            unique_vals = sorted(set(sample_mask.tolist()))
        else:
            unique_vals = sorted(set(sample_mask))
        
        logger.info(f"  Unique mask values found: {unique_vals}")
    
    logger.info("\n" + "="*80)


def compare_before_after(
    checkpoint_id: int = 0,
    paper_idx: int = 0,
    checkpoint_dir: str = CHECKPOINT_DIR
) -> None:
    """
    Compare original vs annotated text (before/after tagging).
    
    Shows exact tag format application.
    
    Args:
        checkpoint_id: Checkpoint number
        paper_idx: Paper index in checkpoint
        checkpoint_dir: Path to checkpoint directory
    """
    
    try:
        df = load_checkpoint(checkpoint_id, checkpoint_dir)
    except Exception as e:
        logger.error(f"✗ Error loading checkpoint: {e}")
        return
    
    if paper_idx >= len(df):
        logger.error(f"✗ Paper index {paper_idx} out of range (max: {len(df)-1})")
        return
    
    paper = df.iloc[paper_idx].to_dict()
    
    logger.info("\n" + "="*80)
    logger.info("BEFORE / AFTER COMPARISON")
    logger.info("="*80)
    
    logger.info(f"\n[Original Text] (before tagging):")
    logger.info(f"  {paper.get('original_text', 'N/A')[:300]}...")
    
    logger.info(f"\n[Annotated Text] (after tagging):")
    logger.info(f"  Format: {TAG_FORMAT}")
    logger.info(f"  {paper.get('annotated_text', 'N/A')[:300]}...")
    
    logger.info(f"\n[Tag Format Explanation]:")
    logger.info(f"  <connector type=\"CAUSAL\"> because </connector>")
    logger.info(f"                        ↑ connector type (6 types)")
    logger.info(f"                                        ↑ actual connector word")
    
    logger.info("\n" + "="*80)


def show_all_stats(checkpoint_dir: str = CHECKPOINT_DIR) -> None:
    """
    Show aggregated statistics across all checkpoints.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    """
    
    try:
        stats = get_checkpoint_stats(checkpoint_dir)
    except Exception as e:
        logger.error(f"✗ Error computing statistics: {e}")
        return
    
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE STATISTICS (ALL CHECKPOINTS)")
    logger.info("="*80)
    
    logger.info(f"\n[Overall]:")
    logger.info(f"  Total papers: {stats.get('total_papers', 0):,}")
    logger.info(f"  Total tokens: {stats.get('total_tokens', 0):,}")
    logger.info(f"  Total connectors: {stats.get('total_connectors', 0):,}")
    
    if stats.get('total_papers', 0) > 0:
        logger.info(f"  Avg tokens/paper: {stats.get('avg_tokens_per_paper', 0):.0f}")
        logger.info(f"  Avg connectors/paper: {stats.get('avg_connectors_per_paper', 0):.1f}")
        logger.info(f"  Connector density: {stats.get('connector_density', 0)*100:.2f}%")
    
    # Domain distribution
    if 'domain_distribution' in stats:
        logger.info(f"\n[Domain Distribution]:")
        for domain, count in sorted(stats['domain_distribution'].items()):
            pct = 100 * count / stats.get('total_papers', 1)
            logger.info(f"  {domain}: {count:,} ({pct:.1f}%)")
    
    # Connector type distribution
    if 'connector_type_distribution' in stats:
        logger.info(f"\n[Connector Type Distribution]:")
        total_connectors = stats.get('total_connectors', 1)
        for conn_type, count in sorted(stats['connector_type_distribution'].items(), key=lambda x: -x[1]):
            pct = 100 * count / total_connectors
            logger.info(f"  {conn_type}: {count:,} ({pct:.1f}%)")
    
    logger.info("\n" + "="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Inspect preprocessed checkpoint files"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=int,
        default=0,
        help='Checkpoint ID to inspect'
    )
    parser.add_argument(
        '--paper',
        type=int,
        default=0,
        help='Paper index within checkpoint'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show before/after comparison'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show aggregate statistics'
    )
    parser.add_argument(
        '--mask',
        action='store_true',
        help='Analyze connector masks'
    )
    parser.add_argument(
        '--dir',
        default=CHECKPOINT_DIR,
        help='Checkpoint directory'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("CHECKPOINT INSPECTOR - CRD")
    logger.info("="*80)
    
    if args.stats:
        show_all_stats(args.dir)
    elif args.compare:
        compare_before_after(args.checkpoint, args.paper, args.dir)
    else:
        # Default: show checkpoint overview + paper details
        show_checkpoint_overview(args.checkpoint, args.dir)
        
        try:
            df = load_checkpoint(args.checkpoint, args.dir)
            if len(df) > args.paper:
                paper = df.iloc[args.paper].to_dict()
                show_paper_details(paper, show_preview=True)
        except Exception as e:
            logger.error(f"✗ Error: {e}")


if __name__ == "__main__":
    main()