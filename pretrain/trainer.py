#!/usr/bin/env python3
"""
pretrain/trainer.py

Training loop manager for connector-aware Llama pretraining.
Handles epoch iteration, batch processing, loss computation,
checkpointing with file tracking in tqdm progress bar.

FIXED: Properly extracts filename from sources tuple.
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import glob
import json
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login, Repository
from utils.clear_memory import clear_cuda_memory, print_cuda_memory


class ConnectorTrainer:
    """Manages training loop with connector-aware boosting and checkpoint management"""
    
    def __init__(self, model_wrapper, cfg, hf_repo_id=None, hf_token=None):
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.cfg = cfg
        self.device = model_wrapper.device
        
        # Hugging Face Hub configuration
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.hf_api = None
        self.hf_repo = None
        
        # Checkpoint directory is now the HF repo clone
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.metadata_file = self.checkpoint_dir / "training_metadata.json"
        
        # Initialize HF repository if configured
        if self.hf_repo_id and self.hf_token:
            self._setup_hf_repo()
        else:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
            print(f"Repository mode: Git-based (checkpoint folder is HF repo)")
        print(f"{'='*70}\n")
    
    def _setup_hf_repo(self):
        """Initialize Hugging Face repository as the checkpoint folder"""
        try:
            login(token=self.hf_token, add_to_git_credential=True)
            self.hf_api = HfApi()
            
            try:
                create_repo(
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                    exist_ok=True,
                    private=False,
                    repo_type="model"
                )
                print(f"✓ Hugging Face repository ready: {self.hf_repo_id}")
            except Exception as e:
                print(f"ℹ️ Repository may already exist: {e}")
            
            if self.checkpoint_dir.exists():
                if (self.checkpoint_dir / ".git").exists():
                    print(f"✓ Found existing git repository at {self.checkpoint_dir}")
                    try:
                        subprocess.run(
                            ["git", "pull"],
                            cwd=str(self.checkpoint_dir),
                            check=True,
                            capture_output=True
                        )
                        print("✓ Pulled latest changes from HF Hub")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ Could not pull changes: {e}")
                else:
                    print(f"⚠️ Checkpoint directory exists but is not a git repo, reinitializing...")
                    self._clone_or_init_repo()
            else:
                self._clone_or_init_repo()
            
            self.hf_repo = Repository(
                local_dir=str(self.checkpoint_dir),
                clone_from=self.hf_repo_id,
                token=self.hf_token,
                git_user="training-bot",
                git_email="training@huggingface.co"
            )
            
            print(f"✓ HF Repository initialized at: {self.checkpoint_dir}")
            
        except Exception as e:
            print(f"❌ Error setting up Hugging Face repository: {e}")
            print("⚠️ Continuing with local checkpoints only")
            self.hf_repo = None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _clone_or_init_repo(self):
        """Clone HF repository or initialize new one"""
        try:
            print(f"📥 Cloning repository from HF Hub to {self.checkpoint_dir}")
            
            if self.checkpoint_dir.exists() and not (self.checkpoint_dir / ".git").exists():
                import shutil
                shutil.rmtree(self.checkpoint_dir)
            
            Repository(
                local_dir=str(self.checkpoint_dir),
                clone_from=self.hf_repo_id,
                token=self.hf_token,
                git_user="training-bot",
                git_email="training@huggingface.co"
            )
            
            print(f"✓ Successfully cloned repository")
            
        except Exception as e:
            print(f"ℹ️ Could not clone (repository may be empty): {e}")
            print(f"Creating new local repository...")
            
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=str(self.checkpoint_dir), check=True)
            
            repo_url = f"https://huggingface.co/{self.hf_repo_id}"
            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=str(self.checkpoint_dir),
                check=False
            )
            subprocess.run(
                ["git", "config", "credential.helper", "store"],
                cwd=str(self.checkpoint_dir),
                check=False
            )
            
            print(f"✓ Initialized new git repository")
    
    def save_checkpoint(self, files_processed, epoch, total_files, avg_loss):
        """Save model checkpoint with metadata and push to HF Hub"""
        print(f"\n{'='*70}")
        print(f"SAVING CHECKPOINT - Files: {files_processed}/{total_files}")
        print(f"{'='*70}")
        
        self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)
        
        metadata = {
            "epoch": epoch,
            "files_processed": files_processed,
            "total_files": total_files,
            "last_file_index": files_processed - 1,
            "avg_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "model_name": self.cfg.model_name,
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {self.checkpoint_dir}")
        print(f"✓ Metadata saved")
        print(f"  - Epoch: {epoch}")
        print(f"  - Files processed: {files_processed}")
        print(f"  - Average loss: {avg_loss:.4f}")
        
        if self.hf_repo:
            self._push_to_hf(files_processed)
        
        print(f"{'='*70}\n")
    
    def _push_to_hf(self, files_processed):
        """Push checkpoint to Hugging Face Hub using git"""
        try:
            print("\n📤 Pushing to Hugging Face Hub...")
            
            commit_message = f"Checkpoint: {files_processed} files processed"
            
            self.hf_repo.push_to_hub(
                commit_message=commit_message,
                blocking=True
            )
            
            print(f"✓ Successfully pushed to {self.hf_repo_id}")
            
        except Exception as e:
            print(f"❌ Error pushing to Hugging Face: {e}")
            print("⚠️ Checkpoint saved locally, will retry on next save")
    
    def load_checkpoint(self):
        """Load checkpoint from HF repository"""
        if self.hf_repo:
            try:
                print(f"\n📥 Pulling latest checkpoint from HF Hub...")
                self.hf_repo.git_pull()
                print("✓ Successfully pulled latest checkpoint")
            except Exception as e:
                print(f"⚠️ Could not pull from HF Hub: {e}")
                print("Continuing with local checkpoint...")
        
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
    
    def train_epoch(self, epoch_num, batch_streamer, files_to_process=None, start_file_idx=0):
        """
        Train for one epoch with embedding-level amplification.
        Shows current file being processed in tqdm progress bar.
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        files_processed = start_file_idx
        model_dtype = next(self.model.parameters()).dtype
        
        checkpoint_frequency = getattr(self.cfg, 'checkpoint_frequency', 20)
        
        # Track current file for progress bar
        current_file_name = None
        current_file_idx = start_file_idx
        
        # Create progress bar
        pbar = tqdm(
            batch_streamer,
            desc=f"Epoch {epoch_num}",
            unit="batch",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for tokens, amps, sources in pbar:
            batch_count += 1
            
            # Extract current file information from sources
            # sources is a list of tuples: [(file_idx, sample_idx, pos, filename), ...]
            if sources and len(sources) > 0:
                # Get first element from sources list
                first_source = sources[0]
                
                # Check if it's the new format with filename (4-tuple)
                if len(first_source) == 4:
                    file_idx, sample_idx, pos, fname = first_source
                elif len(first_source) == 3:
                    # Old format without filename
                    file_idx, sample_idx, pos = first_source
                    fname = f"file_{file_idx}"
                else:
                    file_idx = first_source[0] if len(first_source) > 0 else 0
                    fname = f"file_{file_idx}"
                
                # Initialize current_file_name on first batch
                if current_file_name is None:
                    current_file_name = Path(fname).name if isinstance(fname, str) else f"file_{file_idx}"
                    current_file_idx = file_idx
                
                # Update file tracking when file changes
                if file_idx != current_file_idx:
                    current_file_idx = file_idx
                    current_file_name = Path(fname).name if isinstance(fname, str) else f"file_{file_idx}"
                    print(f"\n📂 Processing file {file_idx + start_file_idx}/{files_to_process or '?'}: {fname}")
                
                # Update progress bar with current file
                pbar.set_description(
                    f"Epoch {epoch_num} | File: {current_file_name} ({current_file_idx + start_file_idx}/{files_to_process or '?'})"
                )
            
            # Convert to tensors
            input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
            amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)
            
            # Reshape to 2D
            input_ids = input_ids.unsqueeze(0)
            amp_vec = amp_vec.unsqueeze(0)
            amp_vec = amp_vec.to(dtype=model_dtype)
            
            self.optimizer.zero_grad()
            
            # Get token embeddings and amplify
            embeddings = self.model.get_input_embeddings()(input_ids)
            amp_vec_expanded = amp_vec.unsqueeze(-1)
            amplified_embeddings = embeddings * amp_vec_expanded
            
            # Create labels
            labels = input_ids.clone()
            labels[0, :-1] = input_ids[0, 1:]
            
            # Create attention mask
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
            
            # Update progress bar with loss
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batch_count:.4f}'
            })
            
            # Check if we've moved to a new file for checkpointing
            if sources and len(sources) > 0:
                current_file = sources[0][0]  # file_idx
                
                if current_file > files_processed:
                    files_processed = current_file
                    
                    # Save checkpoint every N files
                    if files_processed % checkpoint_frequency == 0:
                        avg_loss = total_loss / batch_count if batch_count > 0 else 0
                        pbar.close()  # Close progress bar before checkpoint
                        
                        self.save_checkpoint(
                            files_processed=files_processed + start_file_idx,
                            epoch=epoch_num,
                            total_files=files_to_process or (files_processed + start_file_idx),
                            avg_loss=avg_loss
                        )
                        
                        # Recreate progress bar with current state
                        pbar = tqdm(
                            desc=f"Epoch {epoch_num} | File: {current_file_name} ({current_file_idx + start_file_idx}/{files_to_process or '?'})",
                            unit="batch",
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                            initial=batch_count
                        )
            
            # Print sample batches
            if batch_count <= self.cfg.max_batches_to_print:
                pbar.write(f"\n--- Batch {batch_count} ---")
                decoded = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                pbar.write(f"Loss: {loss.item():.4f}")
                pbar.write(f"Amplification (mean): {amp_vec.mean().item():.4f}")
                pbar.write(f"Decoded (first 100 chars): {decoded[:100]}...")
                pbar.write(f"Amps (first 10): {amps[:10]}")
        
        pbar.close()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"\nEpoch {epoch_num} complete - Avg loss: {avg_loss:.4f}\n")
        
        # Save final checkpoint at end of epoch
        self.save_checkpoint(
            files_processed=files_processed + start_file_idx,
            epoch=epoch_num,
            total_files=files_to_process or (files_processed + start_file_idx),
            avg_loss=avg_loss
        )
        
        return avg_loss