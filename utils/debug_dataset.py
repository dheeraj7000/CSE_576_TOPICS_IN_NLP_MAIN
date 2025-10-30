#!/usr/bin/env python3
"""
debug_dataset_comprehensive.py - Deep inspection of dataset content

Analyzes dataset to identify:
- Field structure variations
- Missing/None/empty content patterns
- Data type issues
- Recommendations for code fixes
"""

import logging
from datasets import load_from_disk
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def comprehensive_dataset_analysis(dataset, sample_size: int = 500):
    """
    Deep analysis of dataset structure and content.
    
    Args:
        dataset: HuggingFace dataset
        sample_size: Number of papers to analyze
    """
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE DATASET ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\n[1] DATASET OVERVIEW")
    logger.info(f"  Total papers: {len(dataset):,}")
    logger.info(f"  Columns: {dataset.column_names}")
    
    # Analyze field structure
    field_analysis = defaultdict(lambda: {
        'present': 0,
        'none': 0,
        'empty_str': 0,
        'empty_list': 0,
        'has_content': 0,
        'types': defaultdict(int),
        'avg_length': 0,
        'examples': []
    })
    
    # Domain distribution
    domains = defaultdict(int)
    
    # Analyze samples
    actual_sample_size = min(sample_size, len(dataset))
    stride = max(1, len(dataset) // actual_sample_size)
    
    logger.info(f"\n[2] SAMPLING STRATEGY")
    logger.info(f"  Total papers: {len(dataset):,}")
    logger.info(f"  Sampling: {actual_sample_size} papers (every {stride}th)")
    logger.info(f"  Sample range: 0 to {len(dataset)-1}")
    
    total_lengths = defaultdict(list)
    
    logger.info(f"\n[3] ANALYZING PAPERS...")
    
    for idx in range(0, len(dataset), stride)[:actual_sample_size]:
        paper = dataset[idx]
        
        # Track domain
        domain = paper.get('domain', 'unknown')
        domains[domain] += 1
        
        # Analyze each field
        for field in dataset.column_names:
            value = paper.get(field)
            
            # Track presence
            if value is None:
                field_analysis[field]['none'] += 1
            elif isinstance(value, str):
                if len(value) == 0:
                    field_analysis[field]['empty_str'] += 1
                else:
                    field_analysis[field]['has_content'] += 1
                    total_lengths[field].append(len(value))
                    if len(field_analysis[field]['examples']) < 2:
                        field_analysis[field]['examples'].append(value[:100])
            elif isinstance(value, list):
                if len(value) == 0:
                    field_analysis[field]['empty_list'] += 1
                else:
                    field_analysis[field]['has_content'] += 1
            else:
                field_analysis[field]['has_content'] += 1
            
            field_analysis[field]['types'][type(value).__name__] += 1
            field_analysis[field]['present'] += 1
    
    # Report domain distribution
    logger.info(f"\n[4] DOMAIN DISTRIBUTION (from {actual_sample_size} samples)")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = 100 * count / actual_sample_size
        logger.info(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Report field analysis
    logger.info(f"\n[5] FIELD CONTENT ANALYSIS")
    for field in dataset.column_names:
        info = field_analysis[field]
        logger.info(f"\n  [{field}]")
        logger.info(f"    Present in samples: {info['present']}/{actual_sample_size}")
        logger.info(f"    None values: {info['none']}")
        logger.info(f"    Empty strings: {info['empty_str']}")
        logger.info(f"    Empty lists: {info['empty_list']}")
        logger.info(f"    Has content: {info['has_content']}")
        
        # Type distribution
        if info['types']:
            logger.info(f"    Types found: {dict(info['types'])}")
        
        # Length stats if strings
        if field in total_lengths and total_lengths[field]:
            lengths = total_lengths[field]
            avg = sum(lengths) / len(lengths)
            max_len = max(lengths)
            min_len = min(lengths)
            logger.info(f"    Text length - Min: {min_len}, Avg: {avg:.0f}, Max: {max_len}")
        
        # Examples
        if info['examples']:
            logger.info(f"    Example (first 100 chars):")
            for ex in info['examples']:
                logger.info(f"      {ex}...")
    
    # Generate recommendations
    logger.info(f"\n[6] RECOMMENDATIONS FOR CODE FIXES")
    
    recommendations = []
    
    # Check for text fields
    text_fields = ['article', 'abstract', 'text', 'opinion', 'content', 'full_text', 'body']
    found_text_fields = [f for f in text_fields if f in dataset.column_names]
    
    logger.info(f"\n  Text fields found: {found_text_fields}")
    
    for field in found_text_fields:
        info = field_analysis[field]
        content_pct = 100 * info['has_content'] / info['present'] if info['present'] > 0 else 0
        
        if content_pct == 0:
            recommendations.append(f"⚠ Field '{field}' has NO content in samples - likely wrong field name")
        elif content_pct < 50:
            recommendations.append(f"⚠ Field '{field}' has {content_pct:.0f}% empty/None - may not be primary text")
        else:
            recommendations.append(f"✓ Field '{field}' has {content_pct:.0f}% content - GOOD candidate")
    
    # Check None handling
    none_issues = []
    for field, info in field_analysis.items():
        if info['none'] > 0 and info['present'] > 0:
            none_pct = 100 * info['none'] / info['present']
            if none_pct > 10:
                none_issues.append(f"  {field}: {none_pct:.0f}% None values - needs null check")
    
    if none_issues:
        logger.info(f"\n  None/Null issues found:")
        for issue in none_issues:
            recommendations.append(issue)
    
    # Check for non-string types
    logger.info(f"\n  Type issues to handle:")
    for field, info in field_analysis.items():
        types = set(info['types'].keys())
        if len(types) > 1 or (len(types) == 1 and list(types)[0] != 'NoneType'):
            if 'str' not in types and 'NoneType' not in types:
                logger.info(f"  {field}: Has non-string types: {types}")
                recommendations.append(f"  {field}: Convert {types} to string before processing")
    
    # Final recommendations
    logger.info(f"\n  ACTION ITEMS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    logger.info("\n" + "="*80 + "\n")
    
    return field_analysis, domains, recommendations


def spot_check_problem_indices(dataset, problem_indices: list):
    """
    Deep dive into specific problematic indices.
    
    Args:
        dataset: Dataset
        problem_indices: List of indices to check (e.g., [2876, 40000, 80000])
    """
    
    logger.info("\n" + "="*80)
    logger.info("SPOT CHECK: PROBLEM INDICES")
    logger.info("="*80)
    
    for idx in problem_indices:
        if idx >= len(dataset):
            logger.info(f"\n✗ Index {idx} out of range (max: {len(dataset)-1})")
            continue
        
        logger.info(f"\n[Paper {idx}]")
        paper = dataset[idx]
        
        # Full field dump
        for field, value in paper.items():
            value_type = type(value).__name__
            
            if value is None:
                logger.info(f"  {field}: None")
            elif isinstance(value, str):
                logger.info(f"  {field} ({value_type}): len={len(value)}, preview='{value[:50]}...'")
            elif isinstance(value, list):
                logger.info(f"  {field} ({value_type}): len={len(value)}, items={value[:3]}...")
            else:
                logger.info(f"  {field} ({value_type}): {str(value)[:50]}")
    
    logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    try:
        logger.info("Loading dataset...")
        dataset = load_from_disk("combined_dataset_sample")
        
        # Comprehensive analysis
        field_info, domains, recs = comprehensive_dataset_analysis(dataset, sample_size=500)
        
        # Spot check problem areas
        # Papers that typically cause issues are around boundaries
        problem_indices = [79957, 79958, 79959, 79960]
        spot_check_problem_indices(dataset, problem_indices)
        
        # Save analysis to file
        logger.info("\nSaving analysis report...")
        report = {
            'total_papers': len(dataset),
            'domains': dict(domains),
            'recommendations': recs,
            'fields': {
                field: {
                    'has_content': field_info[field]['has_content'],
                    'none_count': field_info[field]['none'],
                    'empty_count': field_info[field]['empty_str'],
                    'types': dict(field_info[field]['types'])
                }
                for field in dataset.column_names
            }
        }
        
        with open('dataset_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✓ Report saved to dataset_analysis_report.json")
    
    except FileNotFoundError:
        logger.error("Combined dataset not found. Run create_balanced_sample_custom.py first.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)