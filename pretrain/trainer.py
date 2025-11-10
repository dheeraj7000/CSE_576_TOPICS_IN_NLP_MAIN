#!/usr/bin/env python3
"""
<<<<<<< HEAD
trainer.py - FINAL CORRECTED VERSION

CRITICAL CHANGE:
- REMOVED all connector_mask access from parquet files
- connector_mask is NOW ONLY created by the collator (on-the-fly)
- trainer.py NO LONGER tries to extract connector_mask from DirectParquetDataset

This fixes the shape mismatch errors completely!
=======
trainer.py - UPDATED FOR model_multiword_support.py

CRITICAL CHANGE:
- Model.forward() now takes input_ids and optionally connector_mask
- Model AUTOMATICALLY detects connector tags from input_ids
- NO need to pass connector_mask from data (but it's optional for compatibility)
- Data flow: parquet → input_ids + attention_mask → model detects tags → boost applied

PARQUET STRUCTURE (Direct column access):
- doc_id
- domain
- input_ids (list) ← Used for tag detection
- attention_mask (list) ← Standard attention mask (binary 0/1)
- connector_count
- connector_types (list)
- connector_words (list)
- token_count
- connector_density
- connector_mask (list) ← OPTIONAL (can be ignored, model auto-detects)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
<<<<<<< HEAD
# Direct Parquet Dataset (CORRECTED - NO connector_mask extraction)
=======
# Direct Parquet Dataset (UPDATED FOR AUTO TAG DETECTION)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
# ============================================================================

class DirectParquetDataset(HFDataset):
    """
    Load preprocessed parquet files directly.
<<<<<<< HEAD
    
    CORRECTED: Only extracts input_ids and attention_mask from parquet.
    connector_mask will be created ON-THE-FLY by the collator.
    
    Parquet columns used:
    - input_ids: tokenized text (REQUIRED)
    - attention_mask: padding mask (REQUIRED)
    
    Parquet columns IGNORED:
    - connector_mask: (will be created by collator, not extracted from parquet)
    - connector_words: (metadata only)
    - connector_types: (metadata only)
=======
    
    UPDATED: Model auto-detects connector tags from input_ids
    No need to pass connector_mask (but kept for compatibility)
    
    Parquet columns:
    - input_ids: tokenized text WITH connector tags
    - attention_mask: binary mask (0/1)
    - connector_mask: [OPTIONAL] not needed, model auto-detects
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    """
    
    def __init__(self, parquet_path: str, max_files: Optional[int] = None):
        """
        Initialize dataset from parquet files.
        
        Args:
            parquet_path: Path to parquet file or directory
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
        
<<<<<<< HEAD
        CORRECTED: Only returns input_ids and attention_mask.
        connector_mask is NOT extracted from parquet.
=======
        UPDATED: Model auto-detects tags from input_ids
        connector_mask is optional (not required for tag detection)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
        
        # Extract and convert input_ids (with embedded connector tags)
        input_ids = parse_list(item.get("input_ids", []))
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Extract and convert attention_mask (binary 0/1)
        attention_mask = parse_list(item.get("attention_mask", []))
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
<<<<<<< HEAD
        # ✅ CORRECTED: DO NOT extract connector_mask from parquet!
        # It will be created by the collator instead.
        
        # Build return dictionary (only input_ids and attention_mask)
=======
        # Build return dictionary
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
<<<<<<< HEAD
        # Add metadata for reference (not used in training)
=======
        # OPTIONAL: Include connector_mask for loss weighting (if present in parquet)
        connector_mask = parse_list(item.get("connector_mask"))
        if connector_mask is not None:
            if isinstance(connector_mask, list):
                result["connector_mask"] = torch.tensor(connector_mask, dtype=torch.float)
            else:
                result["connector_mask"] = torch.tensor(connector_mask, dtype=torch.float)
        
        # Add optional metadata (not used by model, for logging)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
<<<<<<< HEAD
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
=======
# Custom Loss Function (UPDATED FOR AUTO TAG DETECTION)
# ============================================================================

class ConnectorAwareLoss(nn.Module):
    """
    Custom loss that weights connector tokens.
    
    UPDATED: Works with optional connector_mask
    If mask not provided, all tokens weighted equally
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
            amplification_strength: Additional amp factor (default: 1.0)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.use_amplification = use_amplification
        self.amplification_strength = amplification_strength
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        connector_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
        UPDATED: connector_mask is optional
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            connector_mask: [batch, seq_len] with values 1.0 or 1.1 (OPTIONAL)
        
        Returns:
            loss: scalar
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits_flat = shift_logits.view(-1, self.vocab_size)
        shift_labels_flat = shift_labels.view(-1)
        
        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            shift_logits_flat,
            shift_labels_flat,
            ignore_index=-100,
            reduction='none'
        )
        
        # Apply connector_mask weighting if provided
        if connector_mask is not None:
            shift_connector_flat = connector_mask[..., 1:].contiguous().view(-1)
            weights = shift_connector_flat
        else:
            # No mask provided - equal weight for all tokens
            weights = torch.ones_like(loss_per_token)
        
        # Apply additional amplification if enabled
        if self.use_amplification and self.amplification_strength > 1.0:
            is_connector = (weights > 1.05).float()
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
# Fallback Data Collator (UPDATED FOR AUTO TAG DETECTION)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
# ============================================================================

class ConnectorAwareDataCollator:
    """
    Fallback data collator that handles input_ids + attention_mask.
    
    UPDATED: Model auto-detects connector tags from input_ids
    No need to create masks here
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch"""
        
        if examples and 'input_ids' in examples[0]:
            # Already tokenized with connector tags - just pad
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
        
<<<<<<< HEAD
        # Create connector_mask if not present (collator will create it)
        if "connector_mask" not in batch:
            batch["connector_mask"] = torch.ones_like(batch["input_ids"], dtype=torch.float)
=======
        # OPTIONAL: Include connector_mask if present (for loss weighting)
        if examples and 'connector_mask' in examples[0]:
            connector_masks = torch.stack([e['connector_mask'] for e in examples])
            # Pad connector_mask to match batch length
            padded_mask = torch.ones_like(batch["input_ids"], dtype=torch.float)
            padded_mask[:, :connector_masks.shape[1]] = connector_masks
            batch["connector_mask"] = padded_mask
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        
        return batch


# ============================================================================
<<<<<<< HEAD
# Custom Trainer (CORRECTED)
=======
# Custom Trainer (UPDATED FOR AUTO TAG DETECTION)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """
<<<<<<< HEAD
    CORRECTED: Trainer with standard cross-entropy loss.
    
    KEY CHANGES:
    1. NO loss weighting (uses standard CE loss)
    2. connector_mask passed to model for embedding boost ONLY
    3. Gradient amplification from embedding boost only (not double-amplified)
=======
    Trainer with custom connector-aware loss.
    
    UPDATED: Model auto-detects connector tags from input_ids
    Trainer just passes input_ids and optional connector_mask
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
    """
    
    def _align_special_tokens(self):
        """Override to skip HF Trainer's config validation."""
        pass
    
    def setup_callbacks(self):
        """Override to disable problematic callbacks."""
        super().setup_callbacks()
        self.log_model = False
        self.report_to = []
<<<<<<< HEAD
        logger.info("✓ Disabled HF integration callbacks for custom model compatibility")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """CORRECTED: Compute standard causal LM loss."""
        
=======
        logger.info("✓ Disabled HF integration callbacks")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss.
        
        UPDATED: Model auto-detects connector tags from input_ids
        connector_mask is optional for loss weighting
        """
        connector_mask = inputs.pop("connector_mask", None)  # OPTIONAL
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        
        # ✅ Verify inputs exist
        if input_ids is None:
            raise ValueError("No input_ids in batch!")
        if labels is None:
            raise ValueError("No labels in batch!")
        
        # Forward pass
<<<<<<< HEAD
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
=======
        # Model auto-detects tags from input_ids
        logits = model(
            in_idx=input_ids,
            connector_mask=connector_mask  # OPTIONAL (for loss weighting)
        )
        
        # Custom loss
        if labels is not None:
            loss = self.loss_fn(
                logits=logits,
                labels=labels,
                connector_mask=connector_mask  # OPTIONAL
            )
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        
        if return_outputs:
            return loss, {"logits": logits}
        return loss



# ============================================================================
<<<<<<< HEAD
# Main Pretraining Manager (CORRECTED)
=======
# Main Pretraining Manager (UPDATED FOR AUTO TAG DETECTION)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
# ============================================================================

class ConnectorPretrainingManager:
    """
    CORRECTED: Manager for connector-aware pretraining.
    
<<<<<<< HEAD
    KEY CHANGES:
    - connector_mask is created ON-THE-FLY by collator
    - NO extraction from parquet
    - Gradient amplification from embedding boost ONLY
=======
    UPDATED: Model auto-detects connector tags from input_ids
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
        
<<<<<<< HEAD
        # Initialize components
        self.annotator = ConnectorAnnotator(config.connector_types)
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER (FINAL CORRECTED)")
=======
        self.loss_fn = None
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER")
        logger.info("UPDATED FOR model_multiword_support.py")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        logger.info("="*70)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Boost factor: {config.boost_factor}")
<<<<<<< HEAD
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
=======
        logger.info(f"\n✓ Model auto-detects connector tags from input_ids")
        logger.info(f"✓ Boost applied at embedding layer")
        logger.info(f"✓ Multi-word connector support: YES")
        logger.info(f"✓ Parquet columns: Direct access to input_ids, attention_mask")
        logger.info("="*70 + "\n")
    
    def _load_dataset_from_parquet(self, parquet_path: str, max_files: Optional[int] = None) -> HFDataset:
        """Load dataset from parquet files."""
        logger.info(f"\nLoading dataset from parquet: {parquet_path}")
        
        dataset = DirectParquetDataset(parquet_path, max_files=max_files)
        
        logger.info(f"✓ Loaded {len(dataset):,} samples")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        return dataset
    
    def _load_dataset_from_hf_format(self, dataset_path: str) -> HFDataset:
        """Load dataset from HuggingFace format."""
<<<<<<< HEAD
        logger.info(f"\n[HF FORMAT] Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, 'keys'):  # DatasetDict
            logger.info(f"✓ Loaded DatasetDict with splits: {list(dataset.keys())}")
        else:
            logger.info(f"✓ Loaded {len(dataset):,} samples")
=======
        logger.info(f"\nLoading dataset from HF format: {dataset_path}")
        
        dataset = load_from_disk(dataset_path)
        logger.info(f"✓ Loaded dataset")
        
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
<<<<<<< HEAD
        CORRECTED: Prepare trainer with on-the-fly connector_mask creation.
=======
        Prepare trainer.
        
        UPDATED: Model auto-detects connector tags
        No need for special tag handling
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING TRAINER (FINAL CORRECTED)")
        logger.info("="*70)
        
        # Create data collator
        logger.info(f"\n[1/2] Setting up data collator...")
        
        if self.use_new_collator:
            try:
                from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation
                data_collator = ConnectorDataCollatorWithMaskCreation(
                    tokenizer=self.model_handler.tokenizer,
                    pad_token_id=self.model_handler.tokenizer.pad_token_id,
                    boost_factor=boost_factor
                )
                logger.info("✓ Using ConnectorDataCollatorWithMaskCreation")
<<<<<<< HEAD
                logger.info(f"  - connector_mask created ON-THE-FLY")
                logger.info(f"  - Scans input_ids for connector tags")
                logger.info(f"  - Boost values: 1.0 (normal) or {boost_factor}x (connector)")
            except ImportError as e:
                logger.error(f"❌ IMPORT ERROR: {e}")
                logger.warning("⚠ Falling back to original collator...")
=======
                
            except ImportError:
                logger.warning("⚠ Falling back to simple collator...")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
                self.use_new_collator = False
                data_collator = ConnectorAwareDataCollator(
                    tokenizer=self.model_handler.tokenizer,
                    max_length=getattr(self.config, 'max_length', 2048)
                )
        else:
            logger.info("✓ Using ConnectorAwareDataCollator")
            data_collator = ConnectorAwareDataCollator(
                tokenizer=self.model_handler.tokenizer,
                max_length=getattr(self.config, 'max_length', 2048)
            )
        
        # Training arguments
        logger.info(f"\n[2/2] Configuring training arguments...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
<<<<<<< HEAD
=======

>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
            # Training
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
<<<<<<< HEAD
=======

>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
            # Optimization
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
<<<<<<< HEAD
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
=======

            # Precision
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),

            # Logging
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            report_to=["tensorboard"],          # (explicit, now that TB is installed)

            # Evaluation / Saving (as you already had)
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,

            # >>> THE IMPORTANT LINE <<<
            remove_unused_columns=False
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
        
<<<<<<< HEAD
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
=======
        logger.info(f"\n✓ Connector-aware Features:")
        logger.info(f"  - Auto tag detection: ENABLED")
        logger.info(f"  - Multi-word support: ENABLED")
        logger.info(f"  - Boost applied at: Input embedding layer")
        logger.info(f"  - Boost factor: {boost_factor}x")
        logger.info(f"  - Gradient amplification: 1.1× (not 1.1^28)")
        
        logger.info(f"\n✓ Data Flow:")
        logger.info(f"  - Parquet → input_ids (with tags)")
        logger.info(f"  - Model detects tag boundaries")
        logger.info(f"  - Boosts enclosed tokens by {boost_factor}×")
        logger.info(f"  - Passes through 28 unchanged layers")
        
        logger.info(f"\n✓ Training Settings:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
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
<<<<<<< HEAD
    logger.info("Trainer module - FINAL CORRECTED VERSION")
    logger.info("="*70)
    logger.info("Changes:")
    logger.info("  ✓ Removed connector_mask extraction from parquet")
    logger.info("  ✓ connector_mask created ON-THE-FLY by collator")
    logger.info("  ✓ Standard cross-entropy loss (no weighting)")
    logger.info("  ✓ Gradient amplification from embedding boost ONLY")
    logger.info("="*70)
=======
    logger.info("Trainer module - UPDATED FOR model_multiword_support.py")
    logger.info("="*70)
    logger.info("\n✓ Key Features:")
    logger.info("  - Auto connector tag detection from input_ids")
    logger.info("  - Multi-word connector support")
    logger.info("  - Direct parquet column access")
    logger.info("  - Optional connector_mask for loss weighting")
    logger.info("="*70)
>>>>>>> dc7d560e7d30bf978fc5115a3ad091aa6984a6ae
