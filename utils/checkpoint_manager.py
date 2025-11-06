#!/usr/bin/env python3
"""
checkpoint_manager.py - UPDATED FOR INTEGRATION

Checkpoint Management for Connector-Aware Preprocessing.
Compatible with config_gpt2_cuda.py and data_loader.py.

Key responsibilities:
- Save preprocessed papers with connector masks to parquet format
- Load checkpoints with validation
- Resume from last checkpoint
- Compute aggregate statistics across checkpoints
- Validate connector_mask integrity
"""

import os
import glob
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# Updated imports for integration
try:
    # Try to import from config module
    from utils.config import Config
    config = Config()
    CHECKPOINT_DIR = config.checkpoint_dir
    CHECKPOINT_SIZE = config.checkpoint_size
    SAVE_TEXT_PREVIEW = config.save_text_preview
    PREVIEW_LENGTH = config.preview_length
    CONNECTOR_ATTENTION_WEIGHT = config.connector_attention_weight
except ImportError:
    # Fallback to default values
    CHECKPOINT_DIR = "./data_splits"
    CHECKPOINT_SIZE = 1000
    SAVE_TEXT_PREVIEW = True
    PREVIEW_LENGTH = 1000
    CONNECTOR_ATTENTION_WEIGHT = 1.1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def ensure_checkpoint_dir(checkpoint_dir: str = None):
    """
    Create checkpoint directory if it doesn't exist.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (uses config default if None)
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"✓ Checkpoint directory ready: {checkpoint_dir}")


# ============================================================================
# CHECKPOINT SAVING
# ============================================================================

def save_checkpoint(
    papers: List[Dict],
    checkpoint_id: int,
    checkpoint_dir: str = None
) -> str:
    """
    Save preprocessed papers to checkpoint file (parquet format).
    
    CRITICAL: Preserves connector_mask field for training.
    
    Checkpoint structure (parquet columns):
    - doc_id: Document identifier
    - domain: Source domain (arxiv, pubmed, legal, openwebmath)
    - input_ids: Token IDs
    - attention_mask: Standard attention mask (1/0)
    - connector_mask: CRITICAL - Connector attention weights (1.0/boost_factor)
    - connector_count: Number of connectors in paper
    - connector_types: List of connector type names
    - connector_words: List of actual connector strings
    - token_count: Total tokens
    - original_text: (optional) Raw text preview
    - annotated_text: (optional) Tagged text preview
    
    Args:
        papers: List of preprocessed paper dicts
        checkpoint_id: Checkpoint number
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Path to saved checkpoint file
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    ensure_checkpoint_dir(checkpoint_dir)
    
    if not papers:
        logger.warning("⚠ Empty paper list, skipping save")
        return ""
    
    # Validate connector_mask presence
    missing_mask = [p for p in papers if 'connector_mask' not in p]
    if missing_mask:
        logger.warning(f"⚠ {len(missing_mask)} papers missing connector_mask")
        for p in missing_mask:
            # Default: all 1.0 (no boost)
            p['connector_mask'] = [1.0] * len(p.get('input_ids', []))
    
    # Create dataframe
    df = pd.DataFrame(papers)
    
    # Save to parquet
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{checkpoint_id:04d}.parquet"
    )
    
    df.to_parquet(checkpoint_path, index=False)
    
    logger.info(f"✓ Saved checkpoint {checkpoint_id:04d}: {len(papers)} papers to {checkpoint_path}")
    
    return checkpoint_path


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def load_checkpoint(
    checkpoint_id: int,
    checkpoint_dir: str = None
) -> pd.DataFrame:
    """
    Load a single checkpoint file.
    
    Validates connector_mask field presence.
    
    Args:
        checkpoint_id: Checkpoint number
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        DataFrame with papers
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint_{checkpoint_id:04d}.parquet"
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    df = pd.read_parquet(checkpoint_path)
    
    # Validate connector_mask field
    if 'connector_mask' not in df.columns:
        logger.warning(f"⚠ connector_mask field missing in checkpoint {checkpoint_id}")
        logger.warning("  Adding default mask (1.0 for all tokens)")
        df['connector_mask'] = df['input_ids'].apply(
            lambda ids: [1.0] * len(ids) if isinstance(ids, list) else [1.0]
        )
    
    logger.info(f"✓ Loaded checkpoint {checkpoint_id:04d}: {len(df)} papers")
    
    return df


def load_all_checkpoints(checkpoint_dir: str = None) -> pd.DataFrame:
    """
    Load all checkpoints and concatenate into single dataframe.
    
    Used during training to load complete preprocessed dataset.
    Compatible with PretrainingDataLoader in data_loader.py.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        DataFrame with all papers from all checkpoints
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_*.parquet")
    ))
    
    if not checkpoint_files:
        logger.error(f"✗ No checkpoints found in {checkpoint_dir}")
        raise FileNotFoundError(f"No checkpoints found: {checkpoint_dir}")
    
    logger.info(f"Found {len(checkpoint_files)} checkpoints")
    
    # Load all
    dfs = []
    for checkpoint_file in checkpoint_files:
        try:
            df = pd.read_parquet(checkpoint_file)
            
            # Validate connector_mask
            if 'connector_mask' not in df.columns:
                logger.warning(f"⚠ No connector_mask in {checkpoint_file}, adding default")
                df['connector_mask'] = df['input_ids'].apply(
                    lambda ids: [1.0] * len(ids) if isinstance(ids, list) else [1.0]
                )
            
            dfs.append(df)
            logger.info(f"  ✓ {os.path.basename(checkpoint_file)}: {len(df)} papers")
        except Exception as e:
            logger.warning(f"  ✗ Error loading {checkpoint_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("Could not load any checkpoints")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"\n✓ Combined all checkpoints: {len(combined):,} total papers")
    
    return combined


# ============================================================================
# CHECKPOINT TRACKING
# ============================================================================

def get_last_checkpoint(checkpoint_dir: str = None) -> Tuple[int, int]:
    """
    Get the ID of the last saved checkpoint and total papers processed.
    
    Used for resuming interrupted preprocessing.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
    
    Returns:
        Tuple of (last_checkpoint_id, total_papers_processed)
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    checkpoint_files = sorted(glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_*.parquet")
    ))
    
    if not checkpoint_files:
        return -1, 0
    
    # Get last checkpoint ID
    last_file = checkpoint_files[-1]
    filename = os.path.basename(last_file)
    last_id = int(filename.replace("checkpoint_", "").replace(".parquet", ""))
    
    # Count total papers
    total = 0
    for f in checkpoint_files:
        try:
            df = pd.read_parquet(f)
            total += len(df)
        except Exception:
            continue
    
    return last_id, total


# ============================================================================
# CHECKPOINT STATISTICS
# ============================================================================

def get_checkpoint_stats(checkpoint_dir: str = None) -> Dict:
    """
    Compute aggregate statistics across all checkpoints.
    
    Includes connector-specific metrics.
    
    Returns:
        Dict with statistics:
        - total_papers: Number of papers
        - total_tokens: Total token count
        - total_connectors: Total connector count
        - avg_connectors_per_paper: Average
        - connector_density: Percentage of tokens that are connectors
        - connector_type_distribution: Breakdown by type
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    try:
        df = load_all_checkpoints(checkpoint_dir)
    except Exception as e:
        logger.error(f"Error loading checkpoints: {e}")
        return {}
    
    stats = {
        'total_papers': len(df),
        'total_tokens': int(df['token_count'].sum()) if 'token_count' in df.columns else 0,
        'total_connectors': int(df['connector_count'].sum()) if 'connector_count' in df.columns else 0,
        'avg_tokens_per_paper': float(df['token_count'].mean()) if 'token_count' in df.columns and len(df) > 0 else 0.0,
        'avg_connectors_per_paper': float(df['connector_count'].mean()) if 'connector_count' in df.columns and len(df) > 0 else 0.0,
    }
    
    # Connector density
    if stats['total_tokens'] > 0:
        stats['connector_density'] = stats['total_connectors'] / stats['total_tokens']
    else:
        stats['connector_density'] = 0.0
    
    # Domain distribution
    if 'domain' in df.columns:
        domain_counts = df['domain'].value_counts().to_dict()
        stats['domain_distribution'] = domain_counts
    
    # Connector type distribution
    if 'connector_types' in df.columns:
        all_types = []
        for types_list in df['connector_types']:
            if isinstance(types_list, list):
                all_types.extend(types_list)
        
        if all_types:
            type_dist = {}
            for ct in all_types:
                type_dist[ct] = type_dist.get(ct, 0) + 1
            stats['connector_type_distribution'] = type_dist
    
    return stats


def save_metadata(metadata: Dict, checkpoint_dir: str = None):
    """
    Save preprocessing metadata (statistics, parameters, etc.).
    
    Args:
        metadata: Dict with metadata to save
        checkpoint_dir: Path to checkpoint directory
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    ensure_checkpoint_dir(checkpoint_dir)
    
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Saved metadata to {metadata_path}")


# ============================================================================
# CHECKPOINT STATUS
# ============================================================================

def print_checkpoint_status(checkpoint_dir: str = None):
    """
    Print status of all checkpoints.
    
    Includes validation of connector_mask field.
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    logger.info("\n" + "="*80)
    logger.info("CHECKPOINT STATUS")
    logger.info("="*80)
    
    checkpoint_files = sorted(glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_*.parquet")
    ))
    
    if not checkpoint_files:
        logger.info(f"✗ No checkpoints found in {checkpoint_dir}")
        logger.info("="*80 + "\n")
        return
    
    logger.info(f"\nFound {len(checkpoint_files)} checkpoints:\n")
    
    total_papers = 0
    total_tokens = 0
    total_connectors = 0
    has_connector_mask = True
    
    for checkpoint_file in checkpoint_files:
        try:
            df = pd.read_parquet(checkpoint_file)
            filename = os.path.basename(checkpoint_file)
            
            papers = len(df)
            tokens = int(df['token_count'].sum()) if 'token_count' in df.columns else 0
            connectors = int(df['connector_count'].sum()) if 'connector_count' in df.columns else 0
            
            # Validate connector_mask
            has_mask = 'connector_mask' in df.columns
            has_connector_mask = has_connector_mask and has_mask
            mask_status = "✓" if has_mask else "✗"
            
            logger.info(f"{mask_status} {filename}: {papers:,} papers, {tokens:,} tokens, {connectors:,} connectors")
            
            total_papers += papers
            total_tokens += tokens
            total_connectors += connectors
        
        except Exception as e:
            logger.error(f"✗ Error reading {checkpoint_file}: {e}")
    
    logger.info("\n" + "-"*80)
    logger.info(f"TOTAL: {total_papers:,} papers, {total_tokens:,} tokens, {total_connectors:,} connectors")
    
    if total_papers > 0:
        logger.info(f"  Avg tokens/paper: {total_tokens/total_papers:.0f}")
        logger.info(f"  Avg connectors/paper: {total_connectors/total_papers:.1f}")
        if total_tokens > 0:
            logger.info(f"  Connector density: {total_connectors/total_tokens*100:.2f}%")
    
    # Connector mask validation
    if has_connector_mask:
        logger.info(f"\n✓ All checkpoints have connector_mask field")
    else:
        logger.warning(f"\n⚠ Some checkpoints missing connector_mask field")
    
    # Metadata
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"\n✓ Metadata file present")
        if 'connector_attention_weight' in metadata:
            logger.info(f"  Connector weight: {metadata['connector_attention_weight']}")
    
    logger.info("="*80 + "\n")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_checkpoints(checkpoint_dir: str = None) -> bool:
    """
    Validate all checkpoints for training requirements.
    
    Checks:
    - All checkpoints exist and are readable
    - connector_mask field present
    - connector_count field present
    - connector_types field present
    - input_ids field present
    - attention_mask field present
    
    Returns:
        True if all valid, False otherwise
    """
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    
    try:
        df = load_all_checkpoints(checkpoint_dir)
    except Exception as e:
        logger.error(f"✗ Could not load checkpoints: {e}")
        return False
    
    required_fields = ['input_ids', 'attention_mask']
    recommended_fields = ['connector_mask', 'connector_count', 'connector_types']
    
    missing_required = [f for f in required_fields if f not in df.columns]
    missing_recommended = [f for f in recommended_fields if f not in df.columns]
    
    if missing_required:
        logger.error(f"✗ Missing required fields: {missing_required}")
        return False
    
    if missing_recommended:
        logger.warning(f"⚠ Missing recommended fields: {missing_recommended}")
        logger.warning("  Training will work but connector boosting may not be optimal")
    
    logger.info("✓ All checkpoints valid for training")
    return True


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def get_config_info() -> Dict:
    """
    Get configuration information for checkpoint manager.
    
    Returns:
        Dict with config settings
    """
    try:
        from utils.config import Config
        config = Config()
        return {
            'checkpoint_dir': config.checkpoint_dir,
            'checkpoint_size': config.checkpoint_size,
            'boost_factor': config.boost_factor,
            'model_name': config.model_name,
            'device': config.device
        }
    except ImportError:
        return {
            'checkpoint_dir': CHECKPOINT_DIR,
            'checkpoint_size': CHECKPOINT_SIZE,
            'boost_factor': CONNECTOR_ATTENTION_WEIGHT,
            'model_name': 'unknown',
            'device': 'unknown'
        }


# ============================================================================
# TESTING & VERIFICATION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("CHECKPOINT MANAGER - INTEGRATION TEST")
    logger.info("="*80)
    
    # Show config
    logger.info("\n[Config Info]")
    config_info = get_config_info()
    for key, value in config_info.items():
        logger.info(f"  {key}: {value}")
    
    # Test 1: Directory creation
    logger.info("\n[Test 1] Checkpoint directory")
    ensure_checkpoint_dir()
    logger.info("✓ Directory ready")
    
    # Test 2: Status check
    logger.info("\n[Test 2] Status check")
    print_checkpoint_status()
    
    # Test 3: Stats
    logger.info("\n[Test 3] Compute statistics")
    try:
        stats = get_checkpoint_stats()
        if stats:
            logger.info(f"✓ Statistics computed:")
            logger.info(f"  Papers: {stats.get('total_papers', 0):,}")
            logger.info(f"  Tokens: {stats.get('total_tokens', 0):,}")
            logger.info(f"  Connectors: {stats.get('total_connectors', 0):,}")
            logger.info(f"  Density: {stats.get('connector_density', 0)*100:.2f}%")
        else:
            logger.info("ℹ No checkpoints to compute stats from")
    except Exception as e:
        logger.info(f"ℹ No checkpoints yet: {e}")
    
    # Test 4: Validation
    logger.info("\n[Test 4] Validation")
    try:
        valid = validate_checkpoints()
        if valid:
            logger.info("✓ Checkpoints valid for training")
        else:
            logger.info("⚠ Checkpoints need fixes")
    except FileNotFoundError:
        logger.info("ℹ No checkpoints to validate")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Integration test complete")
    logger.info("="*80 + "\n")
