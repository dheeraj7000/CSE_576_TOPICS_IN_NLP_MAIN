#!/usr/bin/env python3
"""
trainer_FIXED.py - FINAL VERSION WITH connector_mask PASSING

CRITICAL FIXES:
1. ✅ connector_mask extracted from inputs (uncommented)
2. ✅ connector_mask PASSED to model forward()
3. ✅ Uses data_loader_FIXED_V3.py for DirectParquetDataset (no duplication)
4. ✅ Standard cross-entropy loss (Approach 1 - no compounding)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional
from transformers import (
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset, load_from_disk
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ast

logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Connector Annotator
# ============================================================================

class ConnectorAnnotator:
    """Annotates text with connector markup using format: <connector type="TYPE">word</connector>"""
    
    def __init__(self, connector_types: Dict[str, List[str]]):
        """
        Args:
            connector_types: Dict mapping connector types to example words
        """
        self.connector_types = connector_types
        self.patterns = self._build_patterns()
        self.start_token = '<connector type="{type}">'
        self.end_token = '</connector>'
    
    def _build_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for each connector type"""
        patterns = {}
        for conn_type, connectors in self.connector_types.items():
            sorted_connectors = sorted(connectors, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(c) for c in sorted_connectors) + r')\b'
            patterns[conn_type] = re.compile(pattern, re.IGNORECASE)
        return patterns
    
    def annotate(self, text: str) -> str:
        """Annotate text with connector markup"""
        if not text or not isinstance(text, str):
            return text
        
        annotated = text
        for conn_type, pattern in self.patterns.items():
            annotated = pattern.sub(
                lambda m: f'{self.start_token.format(type=conn_type)}{m.group(0)}{self.end_token}',
                annotated
            )
        return annotated


# ============================================================================
# Fallback Data Collator
# ============================================================================

class ConnectorAwareDataCollator:
    """Fallback data collator that creates connector masks"""
    
    def __init__(
        self,
        tokenizer,
        annotator: ConnectorAnnotator,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.annotator = annotator
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with connector masks"""
        
        # Check if examples already have input_ids
        if examples and 'input_ids' in examples[0]:
            # Already tokenized - just pad
            batch = self.tokenizer.pad(
                examples,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            raise ValueError(
                f"Examples must contain 'input_ids'. "
                f"Got keys: {list(examples[0].keys()) if examples else 'empty'}"
            )
        
        # Create labels
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        # Create connector_mask if not present (collator will create it)
        if "connector_mask" not in batch:
            batch["connector_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)
        
        return batch


# ============================================================================
# Custom Trainer (FIXED)
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """
    FIXED: Trainer that passes connector_mask to model.
    
    KEY FEATURES:
    1. ✅ Extracts connector_mask from batch (created by collator)
    2. ✅ Passes connector_mask to model for embedding boost
    3. ✅ Uses standard cross-entropy loss (Approach 1 - no compounding)
    """
    
    def _align_special_tokens(self):
        """Override to skip HF Trainer's config validation."""
        pass
    
    def setup_callbacks(self):
        """Override to disable problematic callbacks for custom models."""
        super().setup_callbacks()
        self.log_model = False
        self.report_to = []
        logger.info("✓ Disabled HF integration callbacks for custom model compatibility")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        FIXED: Compute loss with connector_mask passed to model.
        
        Flow:
        1. Extract connector_mask from inputs (created by collator)
        2. Pass connector_mask to model for 1.1× embedding boost
        3. Compute standard cross-entropy loss (no weighting)
        """
        
        # ✅ FIXED: Extract connector_mask from inputs
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        connector_mask = inputs.get("connector_mask", None)  # ← UNCOMMENTED!
        
        # ✅ FIXED: Forward pass with connector_mask
        logits = model(
            in_idx=input_ids,
            connector_mask=connector_mask  # ← NOW PASSED TO MODEL!
        )
        
        # Standard cross-entropy loss (Approach 1 - no compounding)
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Standard CE loss (no weighting)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        else:
            # Fallback if no labels
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        if return_outputs:
            return loss, {"logits": logits}
        return loss


# ============================================================================
# Main Pretraining Manager (FIXED)
# ============================================================================

class ConnectorPretrainingManager:
    """
    FIXED: Manager for connector-aware pretraining.
    
    Features:
    - Uses data_loader_FIXED_V3.py for DirectParquetDataset
    - Uses ConnectorDataCollatorWithMaskCreation for on-the-fly mask creation
    - Passes connector_mask to model correctly
    - Approach 1: Embedding boost only (no compounding)
    """
    
    def __init__(self, config, model_handler, use_new_collator: bool = True):
        """
        Args:
            config: Config instance
            model_handler: Model handler (from model.py)
            use_new_collator: Use ConnectorDataCollatorWithMaskCreation (recommended: True)
        """
        self.config = config
        self.model_handler = model_handler
        self.use_new_collator = use_new_collator
        
        # Initialize components
        self.annotator = ConnectorAnnotator(config.connector_types)
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER (FINAL FIXED)")
        logger.info("="*70)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Boost factor: {config.boost_factor}")
        logger.info(f"Approach: 1 (Embedding boost only - no compounding)")
        logger.info(f"connector_mask: Created on-the-fly by collator")
        logger.info(f"Loss: Standard cross-entropy (no weighting)")
        if use_new_collator:
            logger.info(f"Data collator: ConnectorDataCollatorWithMaskCreation")
        else:
            logger.info(f"Data collator: ConnectorAwareDataCollator (FALLBACK)")
        logger.info("="*70 + "\n")
    
    def _load_dataset_from_parquet(self, parquet_path: str, max_files: Optional[int] = None) -> HFDataset:
        """
        Load dataset from parquet using data_loader_FIXED_V3.py
        """
        logger.info(f"\n[PARQUET] Loading dataset from: {parquet_path}")
        
        try:
            # ✅ FIXED: Import from data_loader_FIXED_V3.py (no duplication)
            from pretrain.data_loader import DirectParquetDataset
            dataset = DirectParquetDataset(parquet_path, max_files=max_files)
            logger.info(f"✓ Loaded {len(dataset):,} samples from parquet")
            return dataset
        except ImportError as e:
            logger.error(f"❌ Could not import DirectParquetDataset from data_loader: {e}")
            logger.error(f"   Make sure data_loader_FIXED_V3.py is in pretrain/ directory")
            raise
    
    def _load_dataset_from_hf_format(self, dataset_path: str) -> HFDataset:
        """Load dataset from HuggingFace format."""
        logger.info(f"\n[HF FORMAT] Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, 'keys'):  # DatasetDict
            logger.info(f"✓ Loaded DatasetDict with splits: {list(dataset.keys())}")
        else:
            logger.info(f"✓ Loaded {len(dataset):,} samples")
        return dataset
    
    def prepare_trainer(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None,
        output_dir: str = "./output/connector_model",
        boost_factor: float = 1.1,
        num_epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        **kwargs
    ):
        """
        Prepare trainer with corrected connector_mask passing.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING TRAINER (FINAL FIXED)")
        logger.info("="*70)
        
        # Create data collator
        logger.info(f"\n[1/2] Setting up data collator...")
        
        if self.use_new_collator:
            # ✅ FIXED: Import from data_loader_FIXED_V3.py
            try:
                from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation
                data_collator = ConnectorDataCollatorWithMaskCreation(
                    tokenizer=self.model_handler.tokenizer,
                    pad_token_id=self.model_handler.tokenizer.pad_token_id,
                    boost_factor=boost_factor
                )
                logger.info("✓ Using ConnectorDataCollatorWithMaskCreation")
                logger.info(f"  - connector_mask created ON-THE-FLY")
                logger.info(f"  - Scans input_ids for connector tags")
                logger.info(f"  - Validates attention_mask for padding")
                logger.info(f"  - Boost values: 1.0 (normal) or {boost_factor}× (connector)")
            except ImportError as e:
                logger.error(f"❌ IMPORT ERROR: {e}")
                logger.warning("⚠ Falling back to original collator...")
                self.use_new_collator = False
                data_collator = ConnectorAwareDataCollator(
                    tokenizer=self.model_handler.tokenizer,
                    annotator=self.annotator,
                    max_length=getattr(self.config, 'max_length', 2048)
                )
        else:
            logger.info("✓ Using ConnectorAwareDataCollator (FALLBACK)")
            data_collator = ConnectorAwareDataCollator(
                tokenizer=self.model_handler.tokenizer,
                annotator=self.annotator,
                max_length=getattr(self.config, 'max_length', 2048)
            )
        
        # Training arguments
        logger.info(f"\n[2/2] Configuring training arguments...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            # Training
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            # Optimization
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            # Precision
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            # Logging
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            # Evaluation
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            # Saving
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            # Reporting
            report_to="tensorboard",
            run_name="connector_pretrain",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        
        logger.info("✓ Training arguments configured")
        
        # Create trainer
        logger.info("\nCreating trainer...")
        
        self.trainer = ConnectorAwareTrainer(
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
            tokenizer=self.model_handler.tokenizer,
            data_collator=data_collator
        )
        
        logger.info("✓ Trainer created")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING SETUP SUMMARY")
        logger.info("="*70)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Training samples: {len(train_dataset):,}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset):,}")
        
        logger.info(f"\nConnector Boosting (Approach 1):")
        logger.info(f"  Location: Embedding layer ONLY")
        logger.info(f"  Boost factor: {boost_factor}×")
        logger.info(f"  Compounding: NO (single boost)")
        logger.info(f"  Gradient amplification: Indirect (from boosted embeddings)")
        
        logger.info(f"\nData Flow:")
        logger.info(f"  1. Parquet: input_ids + attention_mask")
        logger.info(f"  2. Collator: Creates connector_mask on-the-fly")
        logger.info(f"  3. Trainer: Passes connector_mask to model")
        logger.info(f"  4. Model: Applies 1.1× boost to connector embeddings")
        logger.info(f"  5. Loss: Standard cross-entropy (no weighting)")
        
        logger.info(f"\nLoss Function:")
        logger.info(f"  Type: Standard cross-entropy")
        logger.info(f"  Weighting: NONE (Approach 1)")
        logger.info(f"  Effect: Clean, simple, no compounding")
        
        logger.info(f"\nTraining Settings:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info("="*70 + "\n")
    
    def train(self):
        """Execute training"""
        if self.trainer is None:
            raise ValueError("Trainer not prepared. Call prepare_trainer() first.")
        
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70 + "\n")
        
        self.trainer.train()
        
        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("="*70 + "\n")
    
    def evaluate(self):
        """Evaluate model"""
        if self.trainer is None:
            raise ValueError("Trainer not prepared")
        
        results = self.trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_model(self, output_dir: str = None):
        """Save trained model"""
        if output_dir is None:
            output_dir = self.trainer.args.output_dir + "/final"
        
        logger.info(f"\nSaving model to: {output_dir}")
        self.trainer.save_model(output_dir)
        self.model_handler.tokenizer.save_pretrained(output_dir)
        logger.info("✓ Model saved")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("="*70)
    logger.info("Trainer module - FINAL FIXED VERSION")
    logger.info("="*70)
    logger.info("Fixes:")
    logger.info("  ✅ connector_mask extraction (uncommented)")
    logger.info("  ✅ connector_mask passed to model forward()")
    logger.info("  ✅ Uses data_loader_FIXED_V3.py (no duplication)")
    logger.info("  ✅ Standard cross-entropy loss (Approach 1)")
    logger.info("  ✅ Padding validation enabled")
    logger.info("="*70)
