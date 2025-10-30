#!/usr/bin/env python3
"""
examples.py - CRD UPDATED: Example Usage & Demonstrations

Example Usage Script demonstrating the complete CRD preprocessing pipeline.

Shows:
1. Loading combined multi-domain dataset (4 datasets)
2. Processing single paper: detect → tag → tokenize
3. Connector mask inspection (0.0 padding, 1.0 regular, 1.1 boosted)
4. Checking preprocessing checkpoints
5. Loading processed data for training

IMPORTANT CONCEPTS:
- Tag format: <connector type="X"> word </connector> (exact)
- Connector weight: 1.1x (learnable - model optimizes during training)
- Mask values: 0.0 (padding), 1.0 (regular), 1.1 (connectors)
- Approach: Discourse-Aware Reasoning (CRD), NOT Thought-of-Words (ToW)
"""

import logging
from typing import Dict, List

from transformers import AutoTokenizer

from config import (
    BASE_MODEL,
    CHECKPOINT_DIR,
    CONNECTOR_ATTENTION_WEIGHT,
    SPECIAL_TOKENS,
    TAG_FORMAT,
)

from dataset_loader import (
    load_combined_datasets,
    extract_paper_text,
    should_keep_paper,
    print_dataset_info,
)

from connector_detector import ConnectorDetector, tag_text
from tokenizer_utils import ConnectorTokenizer
from checkpoint_manager import (
    load_checkpoint,
    get_checkpoint_stats,
    print_checkpoint_status,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: LOAD COMBINED DATASET
# ============================================================================

def example_1_load_dataset():
    """
    Example 1: Load combined dataset from all 4 domains.
    
    Demonstrates multi-domain loading for diverse connector patterns.
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Load Combined Dataset")
    logger.info("="*80)
    
    try:
        dataset = load_combined_datasets()
        print_dataset_info(dataset)
    except Exception as e:
        logger.error(f"Error: {e}")


# ============================================================================
# EXAMPLE 2: DETECT & TAG CONNECTORS
# ============================================================================

def example_2_detect_and_tag():
    """
    Example 2: Detect connectors and apply exact tag format.
    
    Shows the tagging process with exact format:
    <connector type="X"> word </connector>
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Detect & Tag Connectors")
    logger.info("="*80)
    
    # Sample text
    text = (
        "The model improves because of better data. However, performance plateaus. "
        "Therefore, we need new approaches. If we use more data, then accuracy increases. "
        "While training, we monitor metrics. Since 2020, progress has been steady."
    )
    
    logger.info(f"\n[Original Text]:\n  {text}")
    
    # Detect and tag
    result = tag_text(text)
    
    logger.info(f"\n[Tagged Text] (Format: {TAG_FORMAT}):\n  {result['tagged_text']}")
    
    # Show analysis
    logger.info(f"\n[Analysis]:")
    logger.info(f"  Total connectors found: {len(result['connector_words'])}")
    logger.info(f"  Connector words: {result['connector_words']}")
    logger.info(f"  Types: {set(result['connector_types'])}")
    
    # Type breakdown
    type_counts = {}
    for ct in result['connector_types']:
        type_counts[ct] = type_counts.get(ct, 0) + 1
    
    logger.info(f"\n  Type distribution:")
    for ctype, count in sorted(type_counts.items()):
        logger.info(f"    {ctype}: {count}")


# ============================================================================
# EXAMPLE 3: TOKENIZE WITH CONNECTOR MASK
# ============================================================================

def example_3_tokenize_with_mask():
    """
    Example 3: Tokenize text and generate connector mask.
    
    Demonstrates mask generation:
    - 0.0 for padding
    - 1.0 for regular tokens
    - 1.1 for connector tokens
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Tokenize with Connector Mask")
    logger.info("="*80)
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    logger.info(f"  Added {num_added} special tokens")
    
    # Create connector tokenizer
    conn_tok = ConnectorTokenizer(tokenizer)
    
    # Sample text with exact tag format
    text = (
        "The model works <connector type=\"CAUSAL\"> because </connector> "
        "of good data. <connector type=\"TEMPORAL\"> When </connector> training, "
        "we monitor loss."
    )
    
    logger.info(f"\n[Input Text]:")
    logger.info(f"  {text}")
    
    # Tokenize with mask
    result = conn_tok.tokenize_with_connector_mask(text, return_tensors="pt")
    
    logger.info(f"\n[Tokenization Result]:")
    logger.info(f"  Tokens: {len(result['input_ids'][0]) if result['input_ids'].dim() > 1 else len(result['input_ids'])}")
    logger.info(f"  Attention mask shape: {result['attention_mask'].shape}")
    logger.info(f"  Connector mask shape: {result['connector_mask'].shape}")
    
    # Analyze mask
    mask = result['connector_mask']
    if mask.dim() > 1:
        mask = mask[0]
    
    mask_list = mask.tolist()
    
    logger.info(f"\n[Connector Mask Analysis]:")
    logger.info(f"  Unique values: {sorted(set(mask_list))}")
    
    padding_count = sum(1 for m in mask_list if m == 0.0)
    regular_count = sum(1 for m in mask_list if m == 1.0)
    boosted_count = sum(1 for m in mask_list if m > 1.05)
    
    logger.info(f"  Padding (0.0): {padding_count} tokens")
    logger.info(f"  Regular (1.0): {regular_count} tokens")
    logger.info(f"  Boosted ({CONNECTOR_ATTENTION_WEIGHT}x): {boosted_count} tokens")
    
    # Show mask pattern
    logger.info(f"\n[Mask Pattern]:")
    logger.info(f"  {mask_list[:30]}...")


# ============================================================================
# EXAMPLE 4: BATCH TOKENIZATION
# ============================================================================

def example_4_batch_tokenization():
    """
    Example 4: Batch tokenization with padding.
    
    Shows how padding is handled (zeroed to 0.0).
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: Batch Tokenization with Padding")
    logger.info("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
    
    conn_tok = ConnectorTokenizer(tokenizer)
    
    # Two texts of different lengths
    texts = [
        "Short text with <connector type=\"CAUSAL\"> because </connector> reason.",
        "This is a much longer text with multiple connectors. It has <connector type=\"TEMPORAL\"> when </connector> needed. "
        "And <connector type=\"ADVERSATIVE\"> but </connector> also challenges. <connector type=\"CONDITIONAL\"> If </connector> trained properly, then good results.",
    ]
    
    logger.info(f"\n[Input Texts]:")
    for i, text in enumerate(texts):
        logger.info(f"  Text {i+1} ({len(text.split())} words): {text[:50]}...")
    
    # Batch tokenize
    batch = conn_tok.tokenize_batch(texts, pad_to_max=True)
    
    logger.info(f"\n[Batch Result]:")
    logger.info(f"  Batch shape: {batch['connector_mask'].shape}")
    logger.info(f"  (batch_size, max_seq_len) = {batch['connector_mask'].shape}")
    
    # Show padding handling
    logger.info(f"\n[Padding Handling]:")
    for i in range(len(texts)):
        mask = batch['connector_mask'][i].tolist()
        padding_zeros = sum(1 for m in mask if m == 0.0)
        logger.info(f"  Sequence {i+1}: {len(mask)} total tokens, {padding_zeros} padding zeros (0.0)")


# ============================================================================
# EXAMPLE 5: INSPECT CHECKPOINTS
# ============================================================================

def example_5_inspect_checkpoints():
    """
    Example 5: Load and inspect preprocessed checkpoints.
    
    Shows checkpoint structure and connector statistics.
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 5: Inspect Preprocessed Checkpoints")
    logger.info("="*80)
    
    # Show checkpoint status
    logger.info(f"\n[Checkpoint Status]:")
    print_checkpoint_status(CHECKPOINT_DIR)
    
    # Try to load first checkpoint
    try:
        logger.info(f"\n[Loading First Checkpoint]:")
        df = load_checkpoint(0, CHECKPOINT_DIR)
        
        logger.info(f"  Papers loaded: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        
        # Show first paper
        if len(df) > 0:
            logger.info(f"\n[First Paper in Checkpoint]:")
            paper = df.iloc[0]
            
            logger.info(f"  Doc ID: {paper.get('doc_id', 'N/A')}")
            logger.info(f"  Domain: {paper.get('domain', 'N/A')}")
            logger.info(f"  Tokens: {paper.get('token_count', 'N/A')}")
            logger.info(f"  Connectors: {paper.get('connector_count', 'N/A')}")
            
            # Check mask
            if 'connector_mask' in paper:
                mask = paper['connector_mask']
                if hasattr(mask, 'tolist'):
                    mask = mask.tolist()
                
                padding = sum(1 for m in mask if m == 0.0) if mask else 0
                regular = sum(1 for m in mask if m == 1.0) if mask else 0
                boosted = sum(1 for m in mask if m > 1.05) if mask else 0
                
                logger.info(f"  Mask - Padding: {padding}, Regular: {regular}, Boosted: {boosted}")
    
    except Exception as e:
        logger.info(f"  (No checkpoints found yet - run preprocess.py first)")


# ============================================================================
# EXAMPLE 6: SHOW STATISTICS
# ============================================================================

def example_6_checkpoint_statistics():
    """
    Example 6: Compute and show aggregate statistics.
    
    Shows connector distribution, domain breakdown, etc.
    """
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 6: Checkpoint Statistics")
    logger.info("="*80)
    
    try:
        stats = get_checkpoint_stats(CHECKPOINT_DIR)
        
        if stats:
            logger.info(f"\n[Aggregate Statistics]:")
            logger.info(f"  Total papers: {stats.get('total_papers', 0):,}")
            logger.info(f"  Total tokens: {stats.get('total_tokens', 0):,}")
            logger.info(f"  Total connectors: {stats.get('total_connectors', 0):,}")
            
            if stats.get('total_papers', 0) > 0:
                logger.info(f"  Avg tokens/paper: {stats.get('avg_tokens_per_paper', 0):.0f}")
                logger.info(f"  Avg connectors/paper: {stats.get('avg_connectors_per_paper', 0):.1f}")
                logger.info(f"  Connector density: {stats.get('connector_density', 0)*100:.2f}%")
            
            # Domain distribution
            if 'domain_distribution' in stats:
                logger.info(f"\n  Domain distribution:")
                for domain, count in sorted(stats['domain_distribution'].items()):
                    logger.info(f"    {domain}: {count:,}")
            
            # Connector type distribution
            if 'connector_type_distribution' in stats:
                logger.info(f"\n  Connector type distribution:")
                for ctype, count in sorted(stats['connector_type_distribution'].items(), key=lambda x: -x[1]):
                    logger.info(f"    {ctype}: {count:,}")
        else:
            logger.info("  (No statistics available - run preprocess.py first)")
    
    except Exception as e:
        logger.info(f"  (No checkpoints to compute stats from)")


# ============================================================================
# KEY CONCEPTS
# ============================================================================

def show_key_concepts():
    """Display key concepts of CRD approach."""
    
    logger.info("\n" + "="*80)
    logger.info("KEY CONCEPTS - DISCOURSE-AWARE REASONING (CRD)")
    logger.info("="*80)
    
    logger.info(f"\n[Tag Format]:")
    logger.info(f"  {TAG_FORMAT}")
    logger.info(f"  Example: <connector type=\"CAUSAL\"> because </connector>")
    
    logger.info(f"\n[Connector Mask Values]:")
    logger.info(f"  0.0  = Padding tokens (explicitly zeroed)")
    logger.info(f"  1.0  = Regular tokens (no boost)")
    logger.info(f"  {CONNECTOR_ATTENTION_WEIGHT}   = Connector tokens (boosted by model during training)")
    
    logger.info(f"\n[Why CRD, not ToW?]:")
    logger.info(f"  ToW: Adds explicit 'Step 1, Step 2' thinking sequence")
    logger.info(f"  CRD: Emphasizes existing discourse connectors in text")
    logger.info(f"  CRD benefits: Less overhead, preserves natural language structure")
    
    logger.info(f"\n[How It Works]:")
    logger.info(f"  1. Detect connectors in text")
    logger.info(f"  2. Tag with exact format: <connector type=\"X\"> word </connector>")
    logger.info(f"  3. Generate connector_mask: 0.0/1.0/1.1 values")
    logger.info(f"  4. Use mask in model attention: scores *= mask (pre-softmax)")
    logger.info(f"  5. Use mask in training loss: loss *= mask")
    logger.info(f"  6. Model learns which connector types matter during training")
    
    logger.info("\n" + "="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    
    logger.info("\n" + "█"*80)
    logger.info("CRD PREPROCESSING PIPELINE - EXAMPLES")
    logger.info("█"*80)
    
    show_key_concepts()
    
    example_1_load_dataset()
    example_2_detect_and_tag()
    example_3_tokenize_with_mask()
    example_4_batch_tokenization()
    example_5_inspect_checkpoints()
    example_6_checkpoint_statistics()
    
    logger.info("\n" + "█"*80)
    logger.info("✓ All examples complete")
    logger.info("█"*80 + "\n")


if __name__ == "__main__":
    main()