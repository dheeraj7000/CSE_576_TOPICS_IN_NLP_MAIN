#!/usr/bin/env python3
"""
connector_training.py (FIXED)

Corrected implementation where:
- Default mask value = 1
- Connector token mask value = 2
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset


# ============================================================================
# Enhanced Connector Annotator (FIXED)
# ============================================================================

class ConnectorAnnotator:
    """Annotates text with connector markup and tracks positions"""
    
    def __init__(self, connector_types: Dict[str, List[str]]):
        """
        Args:
            connector_types: Dict mapping connector types to example words
        """
        self.connector_types = connector_types
        self.patterns = self._build_patterns()
        
        self.start_token = "<connector>"
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
        """Annotate text with connector markup"""
        if not text or not isinstance(text, str):
            return text
        
        annotated = text
        for conn_type, pattern in self.patterns.items():
            annotated = pattern.sub(
                lambda m: f'{self.start_token} type="{conn_type}" {m.group(0)} {self.end_token}',
                annotated
            )
        return annotated
    
    def identify_connector_spans(
        self,
        tokenizer: PreTrainedTokenizer,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Identify which tokens are connector tokens.
        
        FIXED: Mask defaults to 1, connector tokens get 2
        
        Args:
            tokenizer: Tokenizer
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Mask [batch_size, seq_len] where:
            - 1 = regular token (default)
            - 2 = connector token (emphasized)
        """
        batch_size, seq_len = input_ids.shape
        
        # FIXED: Initialize to 1 instead of 0
        connector_mask = torch.ones_like(input_ids, dtype=torch.float)
        
        # Get special token IDs
        start_token_id = tokenizer.convert_tokens_to_ids(self.start_token)
        end_token_id = tokenizer.convert_tokens_to_ids(self.end_token)
        
        # For each sequence in batch
        for batch_idx in range(batch_size):
            in_connector = False
            
            for seq_idx in range(seq_len):
                token_id = input_ids[batch_idx, seq_idx].item()
                
                # Start of connector span
                if token_id == start_token_id:
                    in_connector = True
                    connector_mask[batch_idx, seq_idx] = 2.0  # FIXED: Set to 2
                
                # Inside connector span
                elif in_connector:
                    connector_mask[batch_idx, seq_idx] = 2.0  # FIXED: Set to 2
                    
                    # End of connector span
                    if token_id == end_token_id:
                        in_connector = False
        
        return connector_mask


# ============================================================================
# Custom Loss Function (FIXED)
# ============================================================================

class ConnectorAwareLoss(nn.Module):
    """
    Custom loss that emphasizes connector tokens.
    
    FIXED: Works with mask values of 1 (normal) and 2 (connector)
    """
    
    def __init__(
        self,
        vocab_size: int,
        use_amplification: bool = True,
        amplification_strength: float = 1.2
    ):
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
        
        FIXED: Mask contains 1 (normal) or 2 (connector)
        
        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            connector_mask: [batch, seq_len] with values 1 or 2
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
        
        # FIXED: Use mask values directly as weights
        # connector_mask is already 1 for normal, 2 for connectors
        weights = shift_connector_flat
        
        # Apply additional amplification if enabled
        if self.use_amplification:
            # Amplify only connector tokens (where mask = 2)
            is_connector = (shift_connector_flat > 1.5).float()  # Detect connectors
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
# Custom Data Collator (UNCHANGED)
# ============================================================================

class ConnectorAwareDataCollator:
    """Data collator that creates connector masks"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        annotator: ConnectorAnnotator,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.annotator = annotator
        self.max_length = max_length
    
    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with connector masks"""
        # Standard padding
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        
        # Identify connector tokens (returns mask with 1s and 2s)
        connector_mask = self.annotator.identify_connector_spans(
            self.tokenizer,
            batch["input_ids"]
        )
        
        batch["connector_mask"] = connector_mask
        
        return batch


# ============================================================================
# Custom Trainer (UNCHANGED)
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """Trainer with custom connector-aware loss"""
    
    def __init__(
        self,
        loss_fn: ConnectorAwareLoss,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override to use custom loss"""
        connector_mask = inputs.pop("connector_mask", None)
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Custom loss
        if labels is not None and connector_mask is not None:
            loss = self.loss_fn(
                logits=logits,
                labels=labels,
                connector_mask=connector_mask
            )
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main Pretraining Manager (UPDATED)
# ============================================================================

class ConnectorPretrainingManager:
    """
    Main class for managing connector-aware pretraining.
    
    FIXED: Now uses correct masking (1 for normal, 2 for connectors)
    """
    
    def __init__(self, config, model_handler):
        """
        Args:
            config: Your Config instance
            model_handler: Your ConnectorModelHandler instance
        """
        self.config = config
        self.model_handler = model_handler
        
        # Initialize components
        self.annotator = ConnectorAnnotator(config.connector_types)
        self.loss_fn = None
        self.trainer = None
        
        print("\n" + "=" * 70)
        print("CONNECTOR PRETRAINING MANAGER (FIXED)")
        print("=" * 70)
        print(f"Model: {config.model_name}")
        print(f"Masking: 1 (default) → 2 (connectors)")
        print("=" * 70)
    
    def prepare_trainer(
        self,
        train_dataset: HFDataset,
        eval_dataset: HFDataset,
        output_dir: str = "./output/connector_model",
        use_amplification: bool = True,
        amplification_strength: float = 1.2,
        num_epochs: int = 2,
        batch_size: int = 2,
        learning_rate: float = 5e-6
    ):
        """
        Prepare trainer with custom loss and data collator.
        
        Args:
            train_dataset: Training dataset (tokenized)
            eval_dataset: Eval dataset (tokenized)
            output_dir: Where to save checkpoints
            use_amplification: Enable additional amplification
            amplification_strength: Additional amplification factor
            num_epochs: Number of epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
        """
        # Create custom loss
        self.loss_fn = ConnectorAwareLoss(
            vocab_size=len(self.model_handler.tokenizer),
            use_amplification=use_amplification,
            amplification_strength=amplification_strength
        )
        
        # Create data collator
        data_collator = ConnectorAwareDataCollator(
            tokenizer=self.model_handler.tokenizer,
            annotator=self.annotator,
            max_length=getattr(self.config, 'max_length', 2048)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=2000,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            
            logging_steps=100,
            logging_dir=f"{output_dir}/logs",
            
            eval_strategy="steps",
            eval_steps=5000,
            
            save_strategy="steps",
            save_steps=10000,
            save_total_limit=3,
            
            report_to="tensorboard",
            run_name="connector_pretrain"
        )
        
        # Create trainer
        self.trainer = ConnectorAwareTrainer(
            loss_fn=self.loss_fn,
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.model_handler.tokenizer,
            data_collator=data_collator
        )
        
        print("\n✓ Trainer prepared")
        print(f"   Base weight (normal tokens): 1x")
        print(f"   Connector weight: 2x")
        if use_amplification:
            print(f"   Additional amplification: {amplification_strength}x")
            print(f"   Effective connector weight: {2.0 * amplification_strength}x")
        else:
            print(f"   Additional amplification: Disabled")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
    
    def train(self):
        """Execute training"""
        if self.trainer is None:
            raise ValueError("Trainer not prepared. Call prepare_trainer() first.")
        
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        self.trainer.train()
        
        print("\n✓ Training complete")
    
    def save_model(self, output_dir: str = None):
        """Save trained model"""
        if output_dir is None:
            output_dir = self.trainer.args.output_dir + "/final"
        
        print(f"\nSaving model to: {output_dir}")
        self.trainer.save_model(output_dir)
        self.model_handler.tokenizer.save_pretrained(output_dir)
        print("✓ Model saved")
