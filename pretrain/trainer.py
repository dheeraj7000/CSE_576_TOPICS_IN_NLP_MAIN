#!/usr/bin/env python3
"""
pretrain/trainer.py

Training loop manager for connector-aware Llama pretraining.
Handles epoch iteration, batch processing, and loss computation.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import glob
from utils.clear_memory import clear_cuda_memory, print_cuda_memory

class ConnectorTrainer:
    """Manages training loop with connector-aware boosting"""
    
    def __init__(self, model_wrapper, cfg):
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.cfg = cfg
        self.device = model_wrapper.device
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate if hasattr(cfg, 'learning_rate') else 5e-5,
            weight_decay=0.01
        )
        
        print(f"\n{'='*70}")
        print("TRAINER INITIALIZATION")
        print(f"{'='*70}")
        print(f"Optimizer: AdamW")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
    
    def train_epoch(self, epoch_num, batch_streamer):
        """
        Train for one epoch using streaming batches.
        
        Args:
            epoch_num: Current epoch number
            batch_streamer: Generator yielding (tokens, amps, sources)
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch_num}")
        print(f"{'='*70}\n")
        
        for tokens, amps, sources in tqdm(batch_streamer, desc=f"Epoch {epoch_num}", unit="batch"):
            batch_count += 1
            
            # Convert to tensors
            input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
            amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)

            input_ids, amp_vec = self._prepare_inputs(
                torch.tensor(tokens, dtype=torch.long),
                torch.tensor(amps, dtype=torch.float32)
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get embeddings and boost
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            amp_vec_expanded = amp_vec.unsqueeze(-1)
            inputs_embeds = inputs_embeds * amp_vec_expanded
            
            # Create labels (shift for next-token prediction)
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100
            
            # Forward
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs.loss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print sample batches
            if batch_count <= self.cfg.max_batches_to_print:
                decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
                print(f"\n--- Batch {batch_count} ---")
                print(f"Loss: {loss.item():.4f}")
                print(f"Decoded (first 100 chars): {decoded[:100]}...")
                print(f"Amps (first 10): {amps[:10]}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"\n{'='*70}")
        print(f"Epoch {epoch_num} Complete")
        print(f"Batches processed: {batch_count}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"{'='*70}\n")

        clear_cuda_memory()
        
        return avg_loss
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        print(f"\nSaving checkpoint to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Checkpoint saved\n")

    def _prepare_inputs(self, input_ids, amp_vec):
        """Move inputs to correct device and dtype"""
        input_ids = input_ids.to(self.device)
        amp_vec = amp_vec.to(self.device)
        
        # Convert to model's dtype (important!)
        if self.model.dtype != torch.long:  # Don't convert token IDs
            amp_vec = amp_vec.to(self.model.dtype)
        
        return input_ids, amp_vec
