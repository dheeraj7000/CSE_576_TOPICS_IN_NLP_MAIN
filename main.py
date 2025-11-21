#!/usr/bin/env python3
"""
pretrain/main.py - CHUNKED VERSION

Entry point with chunked batch counting.
Processes files in chunks (e.g., 5 at a time) with accurate progress per chunk.
"""

import glob
import torch
import argparse
import os
from pathlib import Path
from utils.config import Config
from pretrain.model import ConnectorAwareModel
from pretrain.trainer import ConnectorTrainer
from utils.clear_memory import clear_cuda_memory, print_cuda_memory
# FIX: Import snapshot_download to avoid git-lfs requirements
from huggingface_hub import HfApi, create_repo, snapshot_download


def setup_checkpoint_repository(checkpoint_dir, hf_repo_id, hf_token):
    """
    Download checkpoint files using snapshot_download.
    This is safer than 'git clone' because it doesn't require git-lfs.
    """
    checkpoint_path = Path(checkpoint_dir)
    
    print("\n" + "="*70)
    print("CHECKPOINT REPOSITORY SETUP")
    print("="*70)
    
    try:
        # Ensure the repo exists on the Hub
        create_repo(repo_id=hf_repo_id, token=hf_token, exist_ok=True, private=False, repo_type="model")
        print(f"✓ HF repository confirmed: {hf_repo_id}")
        
        # Download the files (Robust method)
        print(f"\n📥 Downloading existing checkpoint files...")
        try:
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=str(checkpoint_path),
                token=hf_token,
                ignore_patterns=[".git", ".git/*"] # Don't need git internals
            )
            print(f"✓ Successfully downloaded checkpoint files")
            return True
        except Exception as e:
            print(f"ℹ️ Download warning (might be empty repo): {e}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            return True
    
    except Exception as e:
        print(f"\n❌ Error setting up repo: {e}")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return False
    finally:
        print("="*70 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Connector-Aware Llama 3.2 Pretraining')
    
    parser.add_argument('--resume-training', action='store_true',
                       help='Resume training from the last checkpoint')
    parser.add_argument('--hf-repo-id', type=str, default="NeuralNinjasConnector/Connector-Llama",
                       help='Hugging Face repository ID')
    parser.add_argument('--hf-token', type=str, default=os.environ.get('HF_TOKEN'),
                       help='Hugging Face API token')
    parser.add_argument('--num-epochs', type=int, default=2,
                       help='Number of training epochs')
    
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        
        print("\n" + "="*70)
        print("CONNECTOR-AWARE LLAMA 3.2 PRETRAINING - CHUNKED MODE")
        print("="*70)
        
        cfg = Config()
        cfg.print_summary()
        
        print_cuda_memory()
        
        # Setup checkpoint repository (Download existing files)
        if args.hf_repo_id and args.hf_token:
            setup_checkpoint_repository(cfg.checkpoint_dir, args.hf_repo_id, args.hf_token)
        
        model_wrapper = None
        metadata = None
        start_file_idx = 0
        start_epoch = 1
        
        if args.resume_training:
            print("\n" + "="*70)
            print("RESUME TRAINING MODE")
            print("="*70)
            
            # Initialize temporary trainer to load metadata
            temp_model = ConnectorAwareModel(cfg)
            temp_trainer = ConnectorTrainer(temp_model, cfg, hf_repo_id=args.hf_repo_id, hf_token=args.hf_token)
            
            # Load the metadata we just downloaded
            metadata = temp_trainer.load_checkpoint()
            
            if metadata:
                start_file_idx = metadata.get('files_processed', 0)
                start_epoch = metadata.get('epoch', 1)
                
                print(f"\n✓ Resuming from checkpoint:")
                print(f"  - Starting epoch: {start_epoch}")
                print(f"  - Files processed: {start_file_idx}")
                
                # Reuse the model we just loaded
                model_wrapper = temp_model
            else:
                print("⚠️ No metadata found. Starting fresh.")
                args.resume_training = False
                model_wrapper = temp_model
        
        if not args.resume_training and model_wrapper is None:
            print("\n" + "="*70)
            print("FRESH TRAINING MODE")
            print("="*70)
            model_wrapper = ConnectorAwareModel(cfg)
        
        print(f"Model parameters: {model_wrapper.get_num_parameters():,}")
        print_cuda_memory()
        
        # Initialize trainer
        trainer = ConnectorTrainer(model_wrapper, cfg, hf_repo_id=args.hf_repo_id, hf_token=args.hf_token)
        
        # Get connector tag IDs
        open_tags = cfg.get_special_tokens()[:-1]
        close_tag = cfg.closing_tag
        tokenizer = model_wrapper.tokenizer
        
        open_tag_ids = set(tokenizer.convert_tokens_to_ids(t) for t in open_tags)
        close_tag_id = tokenizer.convert_tokens_to_ids(close_tag)
        
        # Get training files
        train_files = sorted(glob.glob(f"{cfg.data_dir}/{cfg.train_pattern}"))
        print(f"\nFound {len(train_files)} training files")
        print(f"Files per chunk: {cfg.files_per_chunk}")
        print(f"Total chunks: {(len(train_files) + cfg.files_per_chunk - 1) // cfg.files_per_chunk}")
        
        # Filter files if resuming
        if args.resume_training and start_file_idx > 0:
            train_files = train_files[start_file_idx:]
            print(f"Resuming from file index {start_file_idx}")
            print(f"Remaining files: {len(train_files)}")
        
        # Training loop
        for epoch in range(start_epoch, args.num_epochs + 1):
            try:
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{args.num_epochs}")
                print(f"{'='*70}\n")
                
                print_cuda_memory()
                
                # Train epoch with chunked processing
                avg_loss = trainer.train_epoch_chunked(
                    epoch_num=epoch,
                    all_train_files=train_files,
                    files_per_chunk=cfg.files_per_chunk,
                    batch_size=cfg.batch_size,
                    open_tag_ids=open_tag_ids,
                    close_tag_id=close_tag_id,
                    boost=cfg.boost_factor,
                    files_to_process=len(train_files),
                    start_file_idx=start_file_idx if epoch == start_epoch else 0
                )
                
                if epoch == start_epoch:
                    start_file_idx = 0
                
                clear_cuda_memory()
                
            except Exception as e:
                print(f"\n❌ Error during epoch {epoch}: {str(e)}")
                clear_cuda_memory()
                raise
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print_cuda_memory()
        
        clear_cuda_memory()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        clear_cuda_memory()
        
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        clear_cuda_memory()
        raise


if __name__ == "__main__":
    main()