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
        """Train for one epoch with embedding-level amplification"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        model_dtype = next(self.model.parameters()).dtype
        
        for tokens, amps, sources in tqdm(batch_streamer, desc=f"Epoch {epoch_num}", unit="batch"):
            batch_count += 1
            
            # Convert to tensors
            input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
            amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)
            
            # ✅ FIX: Reshape to 2D [1, seq_len] (batch_size=1)
            input_ids = input_ids.unsqueeze(0)  # [seq_len] → [1, seq_len]
            amp_vec = amp_vec.unsqueeze(0)      # [seq_len] → [1, seq_len]
            
            # Convert amp to model dtype
            amp_vec = amp_vec.to(dtype=model_dtype)
            
            self.optimizer.zero_grad()

            # print(f"Tokens from data loader: {len(tokens)}")  # Should print 128, not 16
            # print(f"Input IDs after unsqueeze: {input_ids.shape}")  # Should be [1, 128]
            
            # Get token embeddings
            embeddings = self.model.get_input_embeddings()(input_ids)
            # Now embeddings shape: [1, seq_len, 3072] ← CORRECT!
            
            # print(f"Input IDs shape: {input_ids.shape}")
            # print(f"Embedding shape: {embeddings.shape}")  # Should be [1, 128, 3072]
            
            # Amplify connector token embeddings
            amp_vec_expanded = amp_vec.unsqueeze(-1)  # [1, seq_len, 1]
            amplified_embeddings = embeddings * amp_vec_expanded
            
            # Create labels (shift for next-token prediction)
            labels = input_ids.clone()
            labels[0, :-1] = input_ids[0, 1:]  # Shift within batch dimension
            # labels[0, -1] = -100

            # print(labels)
            
            # Create 2D attention_mask
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            
            # Forward pass
            outputs = self.model(
                inputs_embeds=amplified_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Print sample batches
            if batch_count <= self.cfg.max_batches_to_print:
                # Flatten back to 1D for decoding
                decoded = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                print(f"\n--- Batch {batch_count} ---")
                print(f"Loss: {loss.item():.4f}")
                print(f"Amplification (mean): {amp_vec.mean().item():.4f}")
                print(f"Decoded (first 100 chars): {decoded[:100]}...")
                print(f"Amps (first 10): {amps[:10]}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"\nEpoch {epoch_num} complete - Avg loss: {avg_loss:.4f}\n")
        
        return avg_loss

    # def train_epoch(self, epoch_num, batch_streamer):
    #     """Train for one epoch with embedding-level amplification"""
    #     self.model.train()
    #     total_loss = 0.0
    #     batch_count = 0
        
    #     model_dtype = next(self.model.parameters()).dtype
        
    #     for tokens, amps, sources in tqdm(batch_streamer, desc=f"Epoch {epoch_num}", unit="batch"):
    #         batch_count += 1
            
    #         # Convert to tensors
    #         input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
    #         amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)
            
    #         # Convert amp to model dtype for amplification
    #         amp_vec = amp_vec.to(self.device, dtype=model_dtype)
            
    #         self.optimizer.zero_grad()
            
    #         # ✅ KEY: Get embeddings, amplify them, then forward
    #         # Get token embeddings
    #         embeddings = self.model.get_input_embeddings()(input_ids)

    #         print("Embedding DIM:", embeddings.shape)
            
    #         # Amplify connector token embeddings
    #         # amp_vec shape: [batch_size] → expand to [batch_size, 1] for broadcasting
    #         amp_vec_expanded = amp_vec.unsqueeze(-1)  # [batch_size, 1]
    #         amplified_embeddings = embeddings * amp_vec_expanded  # [batch_size, hidden_dim]
            
    #         # Create labels (shift for next-token prediction)
    #         labels = input_ids.clone()
    #         labels[:-1] = input_ids[1:]
    #         labels[-1] = -100

    #         print(labels)
            
    #        # ✅ KEY FIX: Create attention_mask (all 1s = all real tokens, no padding)
    #         attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)
            
    #         # Forward pass with AMPLIFIED embeddings + attention_mask
    #         outputs = self.model(
    #             inputs_embeds=amplified_embeddings,  # Use amplified embeddings
    #             attention_mask=attention_mask,       # ← ADD THIS!
    #             labels=labels,
    #             return_dict=True
    #         )
            
    #         loss = outputs.loss
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #         self.optimizer.step()
            
    #         total_loss += loss.item()
            
    #         # Print sample batches
    #         if batch_count <= self.cfg.max_batches_to_print:
    #             decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
    #             print(f"\n--- Batch {batch_count} ---")
    #             print(f"Loss: {loss.item():.4f}")
    #             print(f"Amplification (mean): {amp_vec.mean().item():.4f}")
    #             print(f"Decoded (first 100 chars): {decoded[:100]}...")
    #             print(f"Amps (first 10): {amps[:10]}")
        
    #     avg_loss = total_loss / batch_count if batch_count > 0 else 0
    #     print(f"\nEpoch {epoch_num} complete - Avg loss: {avg_loss:.4f}\n")
        
    #     return avg_loss

    
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
