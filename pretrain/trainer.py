#!/usr/bin/env python3
"""
pretrain/trainer.py

Training loop manager for connector-aware Llama pretraining.
Handles epoch iteration, batch processing, loss computation,
checkpointing, and Hugging Face Hub integration.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import glob
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from utils.clear_memory import clear_cuda_memory, print_cuda_memory


class ConnectorTrainer:
    """Manages training loop with connector-aware boosting and checkpoint management"""
    
    def __init__(self, model_wrapper, cfg, hf_repo_id=None, hf_token=None):
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.cfg = cfg
        self.device = model_wrapper.device
        
        # Hugging Face Hub configuration
        self.hf_repo_id = hf_repo_id  # e.g., "username/model-name"
        self.hf_token = hf_token
        self.hf_api = None
        
        if self.hf_repo_id and self.hf_token:
            self._setup_hf_hub()
        
        # Checkpoint configuration
        self.checkpoint_dir = Path(cfg.checkpoint_dir) / "latest"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "training_metadata.json"
        
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
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        if self.hf_repo_id:
            print(f"Hugging Face repository: {self.hf_repo_id}")
        print(f"{'='*70}\n")
    
    def _setup_hf_hub(self):
        """Initialize Hugging Face Hub API and create repository if needed"""
        try:
            # Login to Hugging Face
            login(token=self.hf_token, add_to_git_credential=True)
            self.hf_api = HfApi()
            
            # Create repository if it doesn't exist
            try:
                create_repo(
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                    exist_ok=True,
                    private=False  # Set to True if you want a private repo
                )
                print(f"✓ Hugging Face repository ready: {self.hf_repo_id}")
            except Exception as e:
                print(f"⚠️ Repository may already exist or creation failed: {e}")
                
        except Exception as e:
            print(f"❌ Error setting up Hugging Face Hub: {e}")
            self.hf_api = None
    
    def save_checkpoint(self, files_processed, epoch, total_files, avg_loss):
        """
        Save model checkpoint with metadata
        
        Args:
            files_processed: Number of files processed so far
            epoch: Current epoch number
            total_files: Total number of training files
            avg_loss: Average loss for this checkpoint
        """
        print(f"\n{'='*70}")
        print(f"SAVING CHECKPOINT - Files: {files_processed}/{total_files}")
        print(f"{'='*70}")
        
        # Save model and tokenizer (overwrites previous checkpoint)
        self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)
        
        # Save training metadata
        metadata = {
            "epoch": epoch,
            "files_processed": files_processed,
            "total_files": total_files,
            "last_file_index": files_processed - 1,
            "avg_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "model_name": self.cfg.model_name,
            "timestamp": str(torch.cuda.Event().record() if torch.cuda.is_available() else None)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {self.checkpoint_dir}")
        print(f"✓ Metadata saved")
        print(f"  - Epoch: {epoch}")
        print(f"  - Files processed: {files_processed}")
        print(f"  - Average loss: {avg_loss:.4f}")
        
        # Upload to Hugging Face Hub
        if self.hf_repo_id and self.hf_api:
            self._upload_to_hf()
        
        print(f"{'='*70}\n")
    
    def _upload_to_hf(self):
        """Upload checkpoint to Hugging Face Hub"""
        try:
            print("\n📤 Uploading to Hugging Face Hub...")
            
            # Upload all files in checkpoint directory
            self.hf_api.upload_folder(
                folder_path=str(self.checkpoint_dir),
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                commit_message=f"Checkpoint: {self._get_metadata()['files_processed']} files processed"
            )
            
            print(f"✓ Successfully uploaded to {self.hf_repo_id}")
            
        except Exception as e:
            print(f"❌ Error uploading to Hugging Face: {e}")
    
    def load_checkpoint(self):
        """
        Load checkpoint from local or Hugging Face
        
        Returns:
            dict: Metadata if checkpoint exists, None otherwise
        """
        # Check if we should fetch from Hugging Face
        if self.hf_repo_id and self.hf_api:
            should_fetch = self._should_fetch_from_hf()
            if should_fetch:
                self._download_from_hf()
        
        # Load local checkpoint
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n{'='*70}")
            print("LOADING CHECKPOINT")
            print(f"{'='*70}")
            print(f"✓ Found checkpoint at: {self.checkpoint_dir}")
            print(f"  - Epoch: {metadata['epoch']}")
            print(f"  - Files processed: {metadata['files_processed']}")
            print(f"  - Last loss: {metadata.get('avg_loss', 'N/A')}")
            print(f"{'='*70}\n")
            
            return metadata
        
        return None
    
    def _should_fetch_from_hf(self):
        """
        Check if Hugging Face repository has a newer checkpoint than local
        
        Returns:
            bool: True if should fetch from HF, False otherwise
        """
        try:
            # Check if local checkpoint exists
            if not self.metadata_file.exists():
                print("ℹ️ No local checkpoint found, will fetch from Hugging Face")
                return True
            
            # Get remote metadata
            remote_files = self.hf_api.list_repo_files(
                repo_id=self.hf_repo_id,
                token=self.hf_token
            )
            
            if "training_metadata.json" not in remote_files:
                print("ℹ️ No remote checkpoint found")
                return False
            
            # Download and compare metadata
            from huggingface_hub import hf_hub_download
            remote_metadata_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="training_metadata.json",
                token=self.hf_token,
                cache_dir=None
            )
            
            with open(remote_metadata_path, 'r') as f:
                remote_metadata = json.load(f)
            
            with open(self.metadata_file, 'r') as f:
                local_metadata = json.load(f)
            
            # Compare files processed
            remote_files_processed = remote_metadata.get('files_processed', 0)
            local_files_processed = local_metadata.get('files_processed', 0)
            
            if remote_files_processed > local_files_processed:
                print(f"✓ Remote checkpoint is ahead ({remote_files_processed} vs {local_files_processed} files)")
                return True
            else:
                print(f"ℹ️ Local checkpoint is up to date ({local_files_processed} files)")
                return False
                
        except Exception as e:
            print(f"⚠️ Error checking remote checkpoint: {e}")
            return False
    
    def _download_from_hf(self):
        """Download checkpoint from Hugging Face Hub"""
        try:
            print(f"\n📥 Downloading checkpoint from Hugging Face: {self.hf_repo_id}")
            
            from huggingface_hub import snapshot_download
            
            # Download entire repository
            snapshot_download(
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                local_dir=str(self.checkpoint_dir),
                local_dir_use_symlinks=False
            )
            
            print(f"✓ Successfully downloaded checkpoint from {self.hf_repo_id}")
            
        except Exception as e:
            print(f"❌ Error downloading from Hugging Face: {e}")
    
    def _get_metadata(self):
        """Get current training metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def train_epoch(self, epoch_num, batch_streamer, files_to_process=None, start_file_idx=0):
        """
        Train for one epoch with embedding-level amplification
        
        Args:
            epoch_num: Current epoch number
            batch_streamer: Iterator yielding batches
            files_to_process: Total number of files to process
            start_file_idx: Index of file to start from (for resuming)
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        files_processed = start_file_idx
        model_dtype = next(self.model.parameters()).dtype
        
        # Track files for checkpointing
        batches_per_file = 1  # Adjust based on your data structure
        checkpoint_frequency = 20  # Save every 20 files
        
        for tokens, amps, sources in tqdm(batch_streamer, desc=f"Epoch {epoch_num}", unit="batch"):
            batch_count += 1
            
            # Convert to tensors
            input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
            amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)
            
            # Reshape to 2D [1, seq_len] (batch_size=1)
            input_ids = input_ids.unsqueeze(0)
            amp_vec = amp_vec.unsqueeze(0)
            
            # Convert amp to model dtype
            amp_vec = amp_vec.to(dtype=model_dtype)
            
            self.optimizer.zero_grad()
            
            # Get token embeddings
            embeddings = self.model.get_input_embeddings()(input_ids)
            
            # Amplify connector token embeddings
            amp_vec_expanded = amp_vec.unsqueeze(-1)
            amplified_embeddings = embeddings * amp_vec_expanded
            
            # Create labels (shift for next-token prediction)
            labels = input_ids.clone()
            labels[0, :-1] = input_ids[0, 1:]
            
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
            
            # Determine if we've moved to a new file
            # This is based on the sources tuple (file_idx, sample_idx, pos)
            if sources and len(sources) > 0:
                current_file = sources[0][0]  # Get file index from first token
                
                # Check if we need to checkpoint
                if current_file > files_processed:
                    files_processed = current_file
                    
                    # Save checkpoint every 20 files
                    if files_processed % checkpoint_frequency == 0:
                        avg_loss = total_loss / batch_count if batch_count > 0 else 0
                        self.save_checkpoint(
                            files_processed=files_processed,
                            epoch=epoch_num,
                            total_files=files_to_process or files_processed,
                            avg_loss=avg_loss
                        )
            
            # Print sample batches
            if batch_count <= self.cfg.max_batches_to_print:
                decoded = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                print(f"\n--- Batch {batch_count} ---")
                print(f"Loss: {loss.item():.4f}")
                print(f"Amplification (mean): {amp_vec.mean().item():.4f}")
                print(f"Decoded (first 100 chars): {decoded[:100]}...")
                print(f"Amps (first 10): {amps[:10]}")
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"\nEpoch {epoch_num} complete - Avg loss: {avg_loss:.4f}\n")
        
        # Save final checkpoint at end of epoch
        self.save_checkpoint(
            files_processed=files_processed,
            epoch=epoch_num,
            total_files=files_to_process or files_processed,
            avg_loss=avg_loss
        )
        
        return avg_loss
    
    def _prepare_inputs(self, input_ids, amp_vec):
        """Move inputs to correct device and dtype"""
        input_ids = input_ids.to(self.device)
        amp_vec = amp_vec.to(self.device)
        
        # Convert to model's dtype (important!)
        if self.model.dtype != torch.long:  # Don't convert token IDs
            amp_vec = amp_vec.to(self.model.dtype)
        
        return input_ids, amp_vec
