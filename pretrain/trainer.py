#!/usr/bin/env python3
"""
trainer.py - FULLY ALIGNED - FIXED FOR PARQUET STRUCTURE
USES DIRECT COLUMN ACCESS from parquet files

Parquet structure:
- doc_id
- domain
- input_ids (list)
- attention_mask (list)
- connector_count
- connector_types (list)
- connector_words (list)
- token_count
- connector_density
- connector_mask (list)

This trainer directly accesses these columns without any transformation.
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
# Direct Parquet Dataset (FIXED for direct column access)
# ============================================================================

class DirectParquetDataset(HFDataset):
    """
    Load preprocessed parquet files directly with proper column access.
    
    Parquet columns:
    - input_ids: tokenized text
    - attention_mask: padding mask
    - connector_mask: boosting mask (1.0/1.1)
    - connector_words: list of connector words
    - connector_types: list of connector types
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
            
            # Validate required columns
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
        
        self.total_samples = len(self.data)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get item by index with proper type conversion.
        
        Handles both string representations and actual lists.
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
        
        # Extract connector_mask if present (will be used by data_loader.py collator)
        connector_mask = parse_list(item.get("connector_mask"))
        
        # Build return dictionary
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add optional fields
        if connector_mask is not None:
            if isinstance(connector_mask, list):
                result["connector_mask"] = torch.tensor(connector_mask, dtype=torch.float)
            else:
                result["connector_mask"] = torch.tensor(connector_mask, dtype=torch.float)
        
        # Add other metadata
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
# Enhanced Connector Annotator
# ============================================================================

class ConnectorAnnotator:
    """Annotates text with connector markup using format: <connector type="x">word</connector>"""
    
    def __init__(self, connector_types: Dict[str, List[str]]):
        """
        Args:
            connector_types: Dict mapping connector types to example words
        """
        self.connector_types = connector_types
        self.patterns = self._build_patterns()
        
        self.start_token = "<connector"
        self.end_token = "</connector>"
    
    def _build_patterns(self) -> Dict[str, re.Pattern]:
        """Build regex patterns for each connector type"""
        patterns = {}
        for conn_type, connectors in self.connector_types.items():
            sorted_connectors = sorted(connectors, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(c) for c in sorted_connectors) + r')\b'
            patterns[conn_type] = re.compile(pattern, re.IGNORECASE)
        return patterns
    
    def annotate(self, text: str) -> str:
        """Annotate text with connector markup using format: <connector type="x">word</connector>"""
        if not text or not isinstance(text, str):
            return text
        
        annotated = text
        for conn_type, pattern in self.patterns.items():
            annotated = pattern.sub(
                lambda m: f'{self.start_token} type="{conn_type}">{m.group(0)}{self.end_token}',
                annotated
            )
        return annotated


# ============================================================================
# Custom Loss Function
# ============================================================================

class ConnectorAwareLoss(nn.Module):
    """
    Custom loss that emphasizes connector tokens.
    Works with mask values of 1.0 (normal) and 1.1 (connector)
    """
    
    def __init__(
        self,
        vocab_size: int,
        use_amplification: bool = False,
        amplification_strength: float = 1.0
    ):
        """
        Args:
            vocab_size: Vocabulary size
            use_amplification: Additional amplification (optional)
            amplification_strength: Additional amp factor (default: 1.0 = no extra amp)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.use_amplification = use_amplification
        self.amplification_strength = amplification_strength
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        connector_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            connector_mask: [batch, seq_len] with values 1.0 or 1.1
        
        Returns:
            loss: scalar
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_connector_mask = connector_mask[..., 1:].contiguous()
        
        # Flatten
        shift_logits_flat = shift_logits.view(-1, self.vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        shift_connector_flat = shift_connector_mask.view(-1)
        
        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            shift_logits_flat,
            shift_labels_flat,
            ignore_index=-100,
            reduction='none'
        )
        
        # Use mask values directly as weights
        weights = shift_connector_flat
        
        # Apply additional amplification if enabled (OPTIONAL)
        if self.use_amplification and self.amplification_strength > 1.0:
            is_connector = (shift_connector_flat > 1.05).float()
            amplification = 1.0 + is_connector * (self.amplification_strength - 1.0)
            weights = weights * amplification
        
        # Apply weights to loss
        weighted_loss = loss_per_token * weights
        
        # Mean over valid tokens
        valid_tokens = (shift_labels_flat != -100).float()
        total_loss = (weighted_loss * valid_tokens).sum()
        total_valid = valid_tokens.sum()
        
        return total_loss / (total_valid + 1e-8)


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
        
        # Create dummy connector_mask (all 1.0) if not present
        if "connector_mask" not in batch:
            batch["connector_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)
        
        return batch


# ============================================================================
# Custom Trainer
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """
    Trainer with custom connector-aware loss.
    Works with Llama3Model from model.py
    """
    
    def __init__(
        self,
        loss_fn: ConnectorAwareLoss,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
    
    def _align_special_tokens(self):
        """
        Override to skip HF Trainer's config validation.
        Our custom model handles special tokens correctly.
        """
        pass
    
    def setup_callbacks(self):
        """Override to disable problematic callbacks for custom models."""
        super().setup_callbacks()
        
        self.log_model = False
        self.report_to = []
        
        logger.info("✓ Disabled HF integration callbacks for custom model compatibility")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute standard causal LM loss."""
        connector_mask = inputs.pop("connector_mask", None)
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        
        # Forward pass
        logits = model(
            in_idx=input_ids,
            connector_mask=connector_mask
        )
        
        # Custom loss
        if labels is not None and connector_mask is not None:
            loss = self.loss_fn(
                logits=logits,
                labels=labels,
                connector_mask=connector_mask
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        if return_outputs:
            return loss, {"logits": logits}
        return loss


# ============================================================================
# Main Pretraining Manager
# ============================================================================

class ConnectorPretrainingManager:
    """
    Main class for managing connector-aware pretraining.
    
    FIXED: Directly accesses parquet columns without transformation
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
        self.loss_fn = None
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER (PARQUET FIXED)")
        logger.info("="*70)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Boost factor: {config.boost_factor}")
        logger.info(f"Architectural boosting: ENABLED")
        logger.info(f"TAG FORMAT: <connector type=\"x\">word</connector>")
        logger.info(f"Parquet columns: Direct access to input_ids, attention_mask, connector_mask")
        if use_new_collator:
            logger.info(f"Data collator: ConnectorDataCollatorWithMaskCreation (from data_loader.py)")
        else:
            logger.info(f"Data collator: ConnectorAwareDataCollator (FALLBACK)")
        logger.info("="*70 + "\n")
    
    def _load_dataset_from_parquet(self, parquet_path: str, max_files: Optional[int] = None) -> HFDataset:
        """
        Load dataset directly from parquet files.
        
        Args:
            parquet_path: Path to parquet files
            max_files: Optional limit on number of files
            
        Returns:
            HFDataset compatible dataset
        """
        logger.info(f"\n[PARQUET] Loading dataset from: {parquet_path}")
        
        dataset = DirectParquetDataset(parquet_path, max_files=max_files)
        
        logger.info(f"✓ Loaded {len(dataset):,} samples from parquet")
        return dataset
    
    def _load_dataset_from_hf_format(self, dataset_path: str) -> HFDataset:
        """
        Load dataset from HuggingFace format.
        
        Args:
            dataset_path: Path to HF dataset directory
            
        Returns:
            HFDataset instance
        """
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
        use_amplification: bool = False,
        amplification_strength: float = 1.0,
        boost_factor: float = 1.1,
        num_epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        **kwargs
    ):
        """
        Prepare trainer with custom loss and data collator.
        
        Args:
            train_dataset: Training dataset (from parquet or HF format)
            eval_dataset: Evaluation dataset (optional)
            output_dir: Output directory for checkpoints
            use_amplification: Additional loss amplification
            amplification_strength: Additional amp factor
            boost_factor: Connector boost factor
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING TRAINER")
        logger.info("="*70)
        
        # Create custom loss
        self.loss_fn = ConnectorAwareLoss(
            vocab_size=len(self.model_handler.tokenizer),
            use_amplification=use_amplification,
            amplification_strength=amplification_strength
        )
        logger.info(f"\n[1/3] Created ConnectorAwareLoss")
        logger.info(f"  - vocab_size: {len(self.model_handler.tokenizer):,}")
        logger.info(f"  - additional_amplification: {use_amplification}")
        
        # Create data collator
        logger.info(f"\n[2/3] Setting up data collator...")
        
        if self.use_new_collator:
            # USE ConnectorDataCollatorWithMaskCreation from data_loader.py
            try:
                from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation
                
                data_collator = ConnectorDataCollatorWithMaskCreation(
                    tokenizer=self.model_handler.tokenizer,
                    pad_token_id=self.model_handler.tokenizer.pad_token_id,
                    boost_factor=boost_factor
                )
                logger.info("✓ Using ConnectorDataCollatorWithMaskCreation (from data_loader.py)")
                logger.info(f"  - Masks created on-the-fly")
                logger.info(f"  - Boost values: 1.0 (normal) or {boost_factor} (connector)")
                
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
        logger.info(f"\n[3/3] Configuring training arguments...")
        
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
            loss_fn=self.loss_fn,
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
        logger.info(f"  Location: Architecture (TransformerBlock)")
        logger.info(f"  Boost factor: {boost_factor}x")
        logger.info(f"  Application: Hidden state multiplication")
        logger.info(f"\nData Structure:")
        logger.info(f"  Parquet columns: Direct access (no transformation)")
        logger.info(f"  input_ids: Extracted from parquet")
        logger.info(f"  attention_mask: Extracted from parquet")
        logger.info(f"  connector_mask: Extracted or created on-the-fly")
        logger.info(f"\nLoss Function:")
        logger.info(f"  Base: Cross-entropy")
        logger.info(f"  Weighting: connector_mask values (1.0 or {boost_factor})")
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
    logger.info("Trainer module - PARQUET FIXED VERSION")
    logger.info("="*70)
    logger.info("Direct column access from parquet files:")
    logger.info("  - input_ids")
    logger.info("  - attention_mask")
    logger.info("  - connector_mask")
    logger.info("  - connector_words")
    logger.info("  - connector_types")
    logger.info("="*70)