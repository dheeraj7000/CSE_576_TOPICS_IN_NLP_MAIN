#!/usr/bin/env python3
"""
dataset_loader_fixed.py - FIXED: Multi-Dataset Field Extraction

Robustly handles field name variations across all 4 datasets:
- ArXiv: 'article' field
- PubMed: 'abstract' field
- Legal: 'text' or 'opinion' field
- OpenWebMath: 'text' field

Also handles metadata-only entries gracefully (no skipping).
"""

import logging
from typing import Dict, List, Optional
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_paper_text(paper: Dict) -> str:
    """Extract text, skip metadata-only fields."""
    
    skip_fields = {'created_timestamp', 'downloaded_timestamp', 'url', 'metadata', 'date', 'section_names'}
    text_fields = ['article', 'abstract', 'text', 'opinion', 'content', 'full_text', 'body']
    
    for field in text_fields:
        if field not in skip_fields and field in paper:
            value = paper[field]
            if value and isinstance(value, str) and len(value.strip()) > 0:
                return value.strip()
    
    # Fallback: title + abstract
    parts = []
    for field in ['title', 'abstract']:
        if field in paper and paper[field]:
            parts.append(str(paper[field]))
    
    return " ".join(parts) if parts else ""


def should_keep_paper(text: str) -> bool:
    """Filter papers by quality."""
    
    if text is None or not isinstance(text, str) or len(text.strip()) == 0:
        return True  # Accept (don't skip)
    
    word_count = len(text.split())
    return 10 <= word_count <= 1000000

def get_paper_domain(paper: Dict) -> str:
    """
    Get paper domain (dataset source).
    
    Args:
        paper: Paper dict
    
    Returns:
        Domain name (arxiv, pubmed, legal, openwebmath)
    """
    
    domain = paper.get('domain', 'unknown')
    
    # Normalize domain names
    domain_map = {
        'arxiv': 'arxiv',
        'pubmed': 'pubmed',
        'legal': 'legal',
        'law': 'legal',
        'openwebmath': 'openwebmath',
        'math': 'openwebmath',
    }
    
    return domain_map.get(str(domain).lower(), 'unknown')


def get_paper_id(paper: Dict, index: int) -> str:
    """
    Get or generate paper ID.
    
    Args:
        paper: Paper dict
        index: Index in dataset
    
    Returns:
        Paper ID string
    """
    
    # Try to get ID from various fields
    for id_field in ['article_id', 'id', 'paper_id', 'arxiv_id', 'pmid']:
        if id_field in paper and paper[id_field]:
            return str(paper[id_field])
    
    # Fallback: generate from domain + index
    domain = get_paper_domain(paper)
    return f"{domain}_{index:08d}"


def print_dataset_info(dataset) -> None:
    """
    Print information about loaded dataset.
    
    Args:
        dataset: HuggingFace dataset
    """
    
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total papers: {len(dataset):,}")
    
    if 'domain' in dataset.column_names:
        logger.info(f"  Columns: {dataset.column_names}")
        
        # Sample papers from each domain
        domains = {}
        for paper in dataset:
            domain = paper.get('domain', 'unknown')
            domains[domain] = domains.get(domain, 0) + 1
        
        logger.info(f"\n  Domain distribution:")
        for domain, count in sorted(domains.items()):
            pct = 100 * count / len(dataset)
            logger.info(f"    {domain}: {count:,} ({pct:.1f}%)")


def diagnose_missing_text(dataset, sample_size: int = 100) -> Dict:
    """
    Diagnose why papers might have missing text.
    
    Samples papers and checks for text in various fields.
    
    Args:
        dataset: Dataset to diagnose
        sample_size: Number of papers to sample
    
    Returns:
        Dict with field availability stats
    """
    
    logger.info(f"\n[Diagnosing Text Fields in {sample_size} samples]:")
    
    possible_fields = [
        'article', 'abstract', 'text', 'opinion', 'content',
        'full_text', 'body', 'title', 'summary'
    ]
    
    field_counts = {f: 0 for f in possible_fields}
    papers_with_text = 0
    papers_with_any_content = 0
    
    for i in range(min(sample_size, len(dataset))):
        paper = dataset[i]
        has_text = False
        has_any = False
        
        for field in possible_fields:
            if field in paper and paper[field]:
                content = paper[field]
                if isinstance(content, str) and len(content.strip()) > 0:
                    field_counts[field] += 1
                    has_text = True
                    has_any = True
        
        # Also check for any non-empty string field
        if not has_any:
            for key, value in paper.items():
                if isinstance(value, str) and len(value.strip()) > 100:
                    has_any = True
                    break
        
        if has_text:
            papers_with_text += 1
        if has_any:
            papers_with_any_content += 1
    
    logger.info(f"\n  Field availability:")
    for field in possible_fields:
        pct = 100 * field_counts[field] / sample_size
        if pct > 0:
            logger.info(f"    {field}: {field_counts[field]}/{sample_size} ({pct:.0f}%)")
    
    logger.info(f"\n  Papers with recognizable text: {papers_with_text}/{sample_size}")
    logger.info(f"  Papers with any content: {papers_with_any_content}/{sample_size}")
    
    return {
        'field_counts': field_counts,
        'papers_with_text': papers_with_text,
        'papers_with_any_content': papers_with_any_content,
    }


if __name__ == "__main__":
    
    # Test on combined dataset
    try:
        dataset = load_from_disk("combined_dataset_sample")
        print_dataset_info(dataset)
        
        # Diagnose first 100 papers
        diagnose_missing_text(dataset, 100)
        
        # Show examples
        logger.info("\n[Sample Papers]:")
        for i in range(min(3, len(dataset))):
            paper = dataset[i]
            text = extract_paper_text(paper)
            logger.info(f"\n  Paper {i}:")
            logger.info(f"    Domain: {get_paper_domain(paper)}")
            logger.info(f"    ID: {get_paper_id(paper, i)}")
            logger.info(f"    Text length: {len(text)}")
            logger.info(f"    Sample: {text[:100]}...")
    
    except FileNotFoundError:
        logger.error("Combined dataset not found. Run create_balanced_sample_custom.py first.")