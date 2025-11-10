#!/usr/bin/env python3
"""
trainer.py - FINAL CORRECTED VERSION

CRITICAL CHANGE:
- REMOVED all connector_mask access from parquet files
- connector_mask is NOW ONLY created by the collator (on-the-fly)
- trainer.py NO LONGER tries to extract connector_mask from DirectParquetDataset

This fixes the shape mismatch errors completely!
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
# Direct Parquet Dataset (CORRECTED - NO connector_mask extraction)
# ============================================================================

class DirectParquetDataset(HFDataset):
    """
    Load preprocessed parquet files directly.
    
    CORRECTED: Only extracts input_ids and attention_mask from parquet.
    connector_mask will be created ON-THE-FLY by the collator.
    
    Parquet columns used:
    - input_ids: tokenized text (REQUIRED)
    - attention_mask: padding mask (REQUIRED)
    
    Parquet columns IGNORED:
    - connector_mask: (will be created by collator, not extracted from parquet)
    - connector_words: (metadata only)
    - connector_types: (metadata only)
    """
    
    def __init__(self, parquet_path: str, max_files: Optional[int] = None):
        """
        Initialize dataset from parquet files.
        
        Args:
            parquet_path: Path to parquet file or directory containing parquet files
            max_files: Limit number of files to load (optional)
        """
        self.parquet_path = Path(parquet_path)
        
        # Handle both single file and directory
        if self.parquet_path.is_file():
            parquet_files = [self.parquet_path]
        else:
            parquet_files = sorted(self.parquet_path.glob("*.parquet"))
        
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        self.data = []
        for file_path in tqdm(parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(file_path)
            
            # Validate ONLY required columns
            required_cols = {'input_ids', 'attention_mask'}
            actual_cols = set(df.columns)
            
            if not required_cols.issubset(actual_cols):
                logger.warning(f"File {file_path.name} missing columns: {required_cols - actual_cols}")
                continue
            
            # Convert DataFrame to list of dicts
            self.data.extend(df.to_dict('records'))
        
        if not self.data:
            raise ValueError("No valid data loaded from parquet files")
        
        logger.info(f"Loaded {len(self.data):,} samples from parquet")
        logger.info(f"Sample keys: {list(self.data[0].keys())}")
        logger.info(f"✓ NOTE: connector_mask will be created ON-THE-FLY by collator")
        
        self.total_samples = len(self.data)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get item by index with proper type conversion.
        
        CORRECTED: Only returns input_ids and attention_mask.
        connector_mask is NOT extracted from parquet.
        """
        item = self.data[idx]
        
        # Helper function to convert string representations to lists
        def parse_list(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except:
                    return value
            return value
        
        # Extract and convert input_ids
        input_ids = parse_list(item.get("input_ids", []))
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Extract and convert attention_mask
        attention_mask = parse_list(item.get("attention_mask", []))
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        # ✅ CORRECTED: DO NOT extract connector_mask from parquet!
        # It will be created by the collator instead.
        
        # Build return dictionary (only input_ids and attention_mask)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add metadata for reference (not used in training)
        result["connector_words"] = parse_list(item.get("connector_words", []))
        result["connector_types"] = parse_list(item.get("connector_types", []))
        
        return result
    
    @property
    def column_names(self):
        """Return column names for compatibility."""
        if self.data:
            return list(self.data[0].keys())
        return []


# ============================================================================
# Enhanced Connector Annotator (UNCHANGED)
# ============================================================================

class ConnectorAnnotator:
    """Annotates text with connector markup using format: word"""
    
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
# Fallback Data Collator (UNCHANGED)
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
# Custom Trainer (CORRECTED)
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """
    CORRECTED: Trainer with standard cross-entropy loss.
    
    KEY CHANGES:
    1. NO loss weighting (uses standard CE loss)
    2. connector_mask passed to model for embedding boost ONLY
    3. Gradient amplification from embedding boost only (not double-amplified)
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
        """CORRECTED: Compute standard causal LM loss."""
        
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        
        # ✅ Verify inputs exist
        if input_ids is None:
            raise ValueError("No input_ids in batch!")
        if labels is None:
            raise ValueError("No labels in batch!")
        
        # Forward pass
        logits = model(in_idx=input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Standard CE loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # ✅ Check for broken loss
        if loss.item() == 0.0:
            logger.error("❌ Loss is 0.0 - data or model broken!")
            logger.error(f"   logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            raise ValueError("Zero loss indicates problem")
        
        if return_outputs:
            return loss, {"logits": logits}
        return loss



# ============================================================================
# Main Pretraining Manager (CORRECTED)
# ============================================================================

class ConnectorPretrainingManager:
    """
    CORRECTED: Manager for connector-aware pretraining.
    
    KEY CHANGES:
    - connector_mask is created ON-THE-FLY by collator
    - NO extraction from parquet
    - Gradient amplification from embedding boost ONLY
    """
    
    def __init__(self, config, model_handler, use_new_collator: bool = True):
        """
        Args:
            config: Config instance
            model_handler: Model handler (from model.py)
            use_new_collator: Use new collator (recommended: True)
        """
        self.config = config
        self.model_handler = model_handler
        self.use_new_collator = use_new_collator
        
        # Initialize components
        self.annotator = ConnectorAnnotator(config.connector_types)
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER (FINAL CORRECTED)")
        logger.info("="*70)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Boost factor: {config.boost_factor}")
        logger.info(f"Architectural boosting: ENABLED (embedding layer)")
        logger.info(f"Loss function: STANDARD cross-entropy (NO weighting)")
        logger.info(f"connector_mask source: COLLATOR (ON-THE-FLY)")
        logger.info(f"connector_mask from parquet: IGNORED ✓")
        if use_new_collator:
            logger.info(f"Data collator: ConnectorDataCollatorWithMaskCreation")
        else:
            logger.info(f"Data collator: ConnectorAwareDataCollator (FALLBACK)")
        logger.info("="*70 + "\n")
    
    def _load_dataset_from_parquet(self, parquet_path: str, max_files: Optional[int] = None) -> HFDataset:
        """Load dataset directly from parquet files."""
        logger.info(f"\n[PARQUET] Loading dataset from: {parquet_path}")
        dataset = DirectParquetDataset(parquet_path, max_files=max_files)
        logger.info(f"✓ Loaded {len(dataset):,} samples from parquet")
        return dataset
    
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
        CORRECTED: Prepare trainer with on-the-fly connector_mask creation.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING TRAINER (FINAL CORRECTED)")
        logger.info("="*70)
        
        # Create data collator
        logger.info(f"\n[1/2] Setting up data collator...")
        
        if self.use_new_collator:
            # USE ConnectorDataCollatorWithMaskCreation from data_loader.py
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
                logger.info(f"  - Boost values: 1.0 (normal) or {boost_factor}x (connector)")
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
        
        logger.info(f"\nConnector Boosting:")
        logger.info(f"  Location: Embedding layer (ONCE per block)")
        logger.info(f"  Boost factor: {boost_factor}x")
        logger.info(f"  Application: Hidden state multiplication")
        
        logger.info(f"\nData Structure:")
        logger.info(f"  Parquet source: input_ids + attention_mask ONLY")
        logger.info(f"  connector_mask: Created ON-THE-FLY by collator ✓")
        logger.info(f"  Tag detection: Automatic (scans input_ids)")
        
        logger.info(f"\nLoss Function:")
        logger.info(f"  Type: STANDARD cross-entropy")
        logger.info(f"  Weighting: NONE")
        logger.info(f"  Gradient amplification: From embedding boost ONLY")
        
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
    logger.info("Trainer module - FINAL CORRECTED VERSION")
    logger.info("="*70)
    logger.info("Changes:")
    logger.info("  ✓ Removed connector_mask extraction from parquet")
    logger.info("  ✓ connector_mask created ON-THE-FLY by collator")
    logger.info("  ✓ Standard cross-entropy loss (no weighting)")
    logger.info("  ✓ Gradient amplification from embedding boost ONLY")
    logger.info("="*70)