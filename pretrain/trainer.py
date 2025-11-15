#!/usr/bin/env python3
"""
pretrain/trainer.py

Training loop manager with chunked batch counting.
Now includes local git commit + push workflow using metadata for commit messages.

Can also be run standalone to commit and push existing checkpoints:
    python pretrain/trainer.py
    python pretrain/trainer.py --checkpoint-dir ./checkpoints
    python pretrain/trainer.py --message "Custom commit message"
"""

import torch
from torch.optim import AdamW
from tqdm import tqdm
import glob
import json
import os
import sys
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
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
        
        # Checkpoint directory
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
                        subprocess.run(["git", "pull"], cwd=str(self.checkpoint_dir), check=True, capture_output=True)
                        print("✓ Pulled latest changes from HF Hub")
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ Could not pull changes: {e}")
                else:
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
            print(f"✓ HF Repository initialized")
            
        except Exception as e:
            print(f"❌ Error setting up Hugging Face repository: {e}")
            self.hf_repo = None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _clone_or_init_repo(self):
        """Clone HF repository or initialize new one"""
        try:
            print(f"📥 Cloning repository from HF Hub")
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
            print(f"ℹ️ Could not clone: {e}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=str(self.checkpoint_dir), check=True)
            repo_url = f"https://huggingface.co/{self.hf_repo_id}"
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=str(self.checkpoint_dir), check=False)
            print(f"✓ Initialized new git repository")
    
    def _count_batches_in_chunk(self, files, batch_size):
        """Count total batches in a chunk of files (e.g., 5 files)."""
        print(f"\n{'='*70}")
        print(f"📊 COUNTING BATCHES IN CURRENT CHUNK")
        print(f"{'='*70}")
        print(f"Files in this chunk:")
        for i, fname in enumerate(files, 1):
            print(f"  {i}. {Path(fname).name}")
        print(f"{'='*70}")
        
        total_tokens = 0
        
        for fname in tqdm(files, desc="Counting tokens", unit="file", ncols=80):
            try:
                df = pd.read_parquet(fname)
                chunk_tokens = sum(len(row.input_ids) for row in df.itertuples(index=False))
                total_tokens += chunk_tokens
            except Exception as e:
                print(f"⚠️ Error reading {Path(fname).name}: {e}")
                continue
        
        num_batches = total_tokens // batch_size
        if total_tokens % batch_size > 0:
            num_batches += 1
        
        print(f"\n{'='*70}")
        print(f"✓ Chunk tokens: {total_tokens:,}")
        print(f"✓ Batch size: {batch_size}")
        print(f"✓ Chunk batches: {num_batches:,}")
        print(f"✓ Estimated time: {num_batches / 5.5:.1f} minutes @ 5.5 batch/sec")
        print(f"{'='*70}\n")
        
        return num_batches
    
    def _commit_and_push_to_hf(self):
        """
        Commit all changes locally and push to Hugging Face Hub.
        Uses metadata file to create detailed commit message.
        """
        try:
            # Check if this is a git repo
            if not (self.checkpoint_dir / ".git").exists():
                print("⚠️ Not a git repository, skipping commit")
                return
            
            # Load metadata for commit message
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    meta = json.load(f)
                
                processed_files = meta.get("processed_files", [])
                files_str = ", ".join(processed_files[-5:]) if processed_files else "N/A"
                
                commit_msg = (
                    f"Checkpoint: {meta.get('files_processed', 'N/A')} files | "
                    f"Epoch {meta.get('epoch', 'N/A')} | "
                    f"Loss: {meta.get('avg_loss', 0):.4f} | "
                    f"Recent: {files_str} | "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                commit_msg = f"Checkpoint update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            print("\n📤 Committing and pushing to Hugging Face Hub...")
            print(f"Commit message: {commit_msg[:100]}...")
            
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self.checkpoint_dir),
                check=True,
                capture_output=True
            )
            
            # Commit locally
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(self.checkpoint_dir),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ Committed changes locally")
            elif "nothing to commit" in result.stdout:
                print("ℹ️ No changes to commit")
                return
            else:
                print(f"⚠️ Commit warning: {result.stdout}")
            
            # Push to Hugging Face Hub
            subprocess.run(
                ["git", "push"],
                cwd=str(self.checkpoint_dir),
                check=True,
                capture_output=True
            )
            
            print(f"✓ Successfully pushed to {self.hf_repo_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Git error: {e}")
            if e.stdout:
                print(f"stdout: {e.stdout.decode()}")
            if e.stderr:
                print(f"stderr: {e.stderr.decode()}")
        except Exception as e:
            print(f"❌ Error during commit/push: {e}")
    
    def save_checkpoint(self, files_processed, epoch, total_files, avg_loss, processed_files=None):
        """Save model checkpoint with metadata and push to HF Hub"""
        print(f"\n{'='*70}")
        print(f"SAVING CHECKPOINT - Files: {files_processed}/{total_files}")
        print(f"{'='*70}")
        
        # Save model and tokenizer
        self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "files_processed": files_processed,
            "total_files": total_files,
            "last_file_index": files_processed - 1,
            "avg_loss": avg_loss,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "model_name": self.cfg.model_name,
            "processed_files": processed_files or [],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved to: {self.checkpoint_dir}")
        print(f"✓ Metadata saved")
        print(f"  - Epoch: {epoch}")
        print(f"  - Files processed: {files_processed}")
        print(f"  - Average loss: {avg_loss:.4f}")
        
        # Commit and push to HF Hub (if configured)
        if self.hf_repo:
            self._commit_and_push_to_hf()
        
        print(f"{'='*70}\n")
    
    def load_checkpoint(self):
        """Load checkpoint from HF repository"""
        if self.hf_repo:
            try:
                print(f"\n📥 Pulling latest checkpoint from HF Hub...")
                subprocess.run(
                    ["git", "pull"],
                    cwd=str(self.checkpoint_dir),
                    check=True,
                    capture_output=True
                )
                print("✓ Successfully pulled latest checkpoint")
            except Exception as e:
                print(f"⚠️ Could not pull from HF Hub: {e}")
        
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
    
    def train_epoch_chunked(self, epoch_num, all_train_files, files_per_chunk, batch_size, 
                           open_tag_ids, close_tag_id, boost, 
                           files_to_process=None, start_file_idx=0):
        """Train for one epoch processing files in chunks."""
        from pretrain.data_loader import token_batch_streamer
        
        self.model.train()
        total_loss = 0.0
        global_batch_count = 0
        files_processed = start_file_idx
        processed_file_names = []
        model_dtype = next(self.model.parameters()).dtype
        
        checkpoint_frequency = getattr(self.cfg, 'checkpoint_frequency', 20)
        
        # Process files in chunks
        file_pointer = 0
        chunk_number = 1
        total_chunks = (len(all_train_files) + files_per_chunk - 1) // files_per_chunk
        
        while file_pointer < len(all_train_files):
            chunk_files = all_train_files[file_pointer:file_pointer + files_per_chunk]
            
            if not chunk_files:
                break
            
            print(f"\n{'#'*70}")
            print(f"# CHUNK {chunk_number}/{total_chunks} - Files {file_pointer}/{len(all_train_files)}")
            print(f"{'#'*70}")
            
            chunk_batches = self._count_batches_in_chunk(chunk_files, batch_size)
            
            batch_stream = token_batch_streamer(
                files=chunk_files,
                batch_size=batch_size,
                open_tag_ids=open_tag_ids,
                close_tag_id=close_tag_id,
                boost=boost
            )
            
            current_file_name = None
            current_file_idx = file_pointer
            chunk_batch_count = 0
            
            pbar = tqdm(
                batch_stream,
                desc=f"Epoch {epoch_num} | Chunk {chunk_number}/{total_chunks}",
                unit="batch",
                total=chunk_batches,
                ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            for tokens, amps, sources in pbar:
                global_batch_count += 1
                chunk_batch_count += 1
                
                if sources and len(sources) > 0:
                    first_source = sources[0]
                    
                    if len(first_source) == 4:
                        file_idx, sample_idx, pos, fname = first_source
                    elif len(first_source) == 3:
                        file_idx, sample_idx, pos = first_source
                        fname = f"file_{file_idx}"
                    else:
                        file_idx = first_source[0] if len(first_source) > 0 else 0
                        fname = f"file_{file_idx}"
                    
                    if current_file_name is None:
                        current_file_name = Path(fname).name if isinstance(fname, str) else f"file_{file_idx}"
                    
                    if file_idx != current_file_idx - file_pointer:
                        current_file_idx = file_idx + file_pointer
                        current_file_name = Path(fname).name if isinstance(fname, str) else f"file_{file_idx}"
                        pbar.write(f"\n📂 Processing: {fname}")
                    
                    pbar.set_description(
                        f"Epoch {epoch_num} | Chunk {chunk_number}/{total_chunks} | {current_file_name}"
                    )
                
                input_ids = torch.tensor(tokens, dtype=torch.long).to(self.device)
                amp_vec = torch.tensor(amps, dtype=torch.float32).to(self.device)
                
                input_ids = input_ids.unsqueeze(0)
                amp_vec = amp_vec.unsqueeze(0).to(dtype=model_dtype)
                
                self.optimizer.zero_grad()
                
                embeddings = self.model.get_input_embeddings()(input_ids)
                amp_vec_expanded = amp_vec.unsqueeze(-1)
                amplified_embeddings = embeddings * amp_vec_expanded
                
                labels = input_ids.clone()
                labels[0, :-1] = input_ids[0, 1:]
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                
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
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/global_batch_count:.4f}'
                })
                
                if sources and len(sources) > 0:
                    current_file = sources[0][0] + file_pointer
                    
                    if current_file > files_processed:
                        files_processed = current_file
                        
                        if isinstance(fname, str):
                            processed_file_names.append(Path(fname).name)
                        
                        if files_processed % checkpoint_frequency == 0:
                            avg_loss = total_loss / global_batch_count if global_batch_count > 0 else 0
                            pbar.close()
                            
                            self.save_checkpoint(
                                files_processed=files_processed + start_file_idx,
                                epoch=epoch_num,
                                total_files=files_to_process or (files_processed + start_file_idx),
                                avg_loss=avg_loss,
                                processed_files=processed_file_names[-checkpoint_frequency:]
                            )
                            
                            pbar = tqdm(
                                desc=f"Epoch {epoch_num} | Chunk {chunk_number}/{total_chunks} | {current_file_name}",
                                unit="batch",
                                total=chunk_batches,
                                initial=chunk_batch_count,
                                ncols=120,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                            )
            
            pbar.close()
            
            print(f"\n✓ Chunk {chunk_number}/{total_chunks} complete!")
            print(f"  Batches processed: {chunk_batch_count}")
            print(f"  Chunk avg loss: {total_loss/global_batch_count:.4f}\n")
            
            file_pointer += files_per_chunk
            chunk_number += 1
        
        avg_loss = total_loss / global_batch_count if global_batch_count > 0 else 0
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch_num} COMPLETE")
        print(f"{'='*70}")
        print(f"Total batches: {global_batch_count}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"{'='*70}\n")
        
        self.save_checkpoint(
            files_processed=files_processed + start_file_idx,
            epoch=epoch_num,
            total_files=files_to_process or (files_processed + start_file_idx),
            avg_loss=avg_loss,
            processed_files=processed_file_names
        )
        
        return avg_loss


# ============================================================================
# STANDALONE MODE: Commit and push checkpoint without training
# ============================================================================

def commit_and_push_standalone(checkpoint_dir, custom_message=None):
    """
    Standalone function to commit and push checkpoint changes.
    Used when trainer.py is run directly.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata_file = checkpoint_dir / "training_metadata.json"
    
    if not checkpoint_dir.exists():
        print(f"❌ Error: Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    if not (checkpoint_dir / ".git").exists():
        print(f"❌ Error: {checkpoint_dir} is not a git repository")
        print("Initialize it first with: git clone https://huggingface.co/<username>/<repo>")
        return False
    
    print(f"\n{'='*70}")
    print("COMMIT AND PUSH CHECKPOINT TO HUGGING FACE HUB")
    print(f"{'='*70}")
    print(f"Directory: {checkpoint_dir}")
    
    # Generate commit message
    if custom_message:
        commit_msg = custom_message
    elif metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                meta = json.load(f)
            
            processed_files = meta.get("processed_files", [])
            files_str = ", ".join(processed_files[-5:]) if processed_files else "N/A"
            
            commit_msg = (
                f"Checkpoint: {meta.get('files_processed', 'N/A')} files | "
                f"Epoch {meta.get('epoch', 'N/A')} | "
                f"Loss: {meta.get('avg_loss', 0):.4f} | "
                f"Recent: {files_str} | "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"✓ Using metadata for commit message")
        except Exception as e:
            print(f"⚠️ Could not read metadata: {e}")
            commit_msg = f"Checkpoint update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        commit_msg = f"Checkpoint update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        print(f"ℹ️ No metadata file found, using default message")
    
    print(f"Commit message: {commit_msg[:100]}...")
    
    try:
        # Check for changes
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(checkpoint_dir),
            capture_output=True,
            text=True,
            check=True
        )
        
        if not status_result.stdout.strip():
            print("\nℹ️ No changes to commit")
            print("✓ Checkpoint is already up to date")
            return True
        
        print(f"\n📝 Changes detected:")
        for line in status_result.stdout.strip().split('\n')[:10]:
            print(f"  {line}")
        
        # Stage changes
        print("\n📦 Staging changes...")
        subprocess.run(["git", "add", "-A"], cwd=str(checkpoint_dir), check=True, capture_output=True)
        print("✓ Changes staged")
        
        # Commit locally
        print("\n💾 Committing locally...")
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(checkpoint_dir), check=True, capture_output=True)
        print("✓ Changes committed locally")
        
        # Push to Hugging Face Hub
        print("\n📤 Pushing to Hugging Face Hub...")
        subprocess.run(["git", "push"], cwd=str(checkpoint_dir), check=True, capture_output=True)
        print("✓ Successfully pushed to Hugging Face Hub")
        
        print(f"\n{'='*70}")
        print("✓ CHECKPOINT SYNC COMPLETE")
        print(f"{'='*70}\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Git error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout.decode()}")
        if e.stderr:
            print(f"stderr: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Commit and push checkpoint to Hugging Face Hub (standalone mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pretrain/trainer.py
  python pretrain/trainer.py --checkpoint-dir ./checkpoints
  python pretrain/trainer.py --message "Updated model weights"
        """
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Path to checkpoint directory (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--message', '-m',
        type=str,
        default=None,
        help='Custom commit message (otherwise uses metadata)'
    )
    
    args = parser.parse_args()
    
    # Run commit and push in standalone mode
    success = commit_and_push_standalone(args.checkpoint_dir, args.message)
    
    sys.exit(0 if success else 1)