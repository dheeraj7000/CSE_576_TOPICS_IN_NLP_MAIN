#!/usr/bin/env python3
"""
pretrain/main.py

Entry point for connector-aware Llama 3.2 pretraining.
Integrates config, data loader, model, and trainer with automatic
checkpoint repository setup and smart resuming.
"""

import glob
import torch
import argparse
import os
from pathlib import Path
from utils.config import Config
from pretrain.data_loader import token_batch_streamer
from pretrain.model import ConnectorAwareModel
from pretrain.trainer import ConnectorTrainer
from utils.clear_memory import clear_cuda_memory, print_cuda_memory


def setup_checkpoint_repository(checkpoint_dir, hf_repo_id, hf_token):
    """
    Setup checkpoint directory as HF repository.
    
    - If directory doesn't exist: Clone from HF
    - If directory exists but not a git repo: Initialize and pull from HF
    - If directory exists and is a git repo: Pull latest changes
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        hf_repo_id: HF repository ID (e.g., "username/model-name")
        hf_token: HF authentication token
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    from huggingface_hub import HfApi, create_repo, Repository
    import subprocess
    
    checkpoint_path = Path(checkpoint_dir)
    
    print("\n" + "="*70)
    print("CHECKPOINT REPOSITORY SETUP")
    print("="*70)
    
    try:
        # Ensure HF repository exists
        api = HfApi()
        try:
            create_repo(
                repo_id=hf_repo_id,
                token=hf_token,
                exist_ok=True,
                private=False,
                repo_type="model"
            )
            print(f"✓ HF repository ready: {hf_repo_id}")
        except Exception as e:
            print(f"ℹ️ Repository may already exist: {e}")
        
        # Case 1: Directory doesn't exist - Clone from HF
        if not checkpoint_path.exists():
            print(f"\n📥 Checkpoint directory not found: {checkpoint_path}")
            print(f"Cloning from HF Hub: {hf_repo_id}")
            
            try:
                Repository(
                    local_dir=str(checkpoint_path),
                    clone_from=hf_repo_id,
                    token=hf_token,
                    git_user="training-bot",
                    git_email="training@huggingface.co"
                )
                print(f"✓ Successfully cloned repository")
                print(f"✓ Checkpoint directory created: {checkpoint_path}")
                return True
                
            except Exception as e:
                print(f"ℹ️ Clone failed (repository may be empty): {e}")
                print(f"Creating new repository...")
                
                # Create directory and initialize
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                
                # Initialize git
                subprocess.run(["git", "init"], cwd=str(checkpoint_path), check=True)
                
                # Add HF remote
                repo_url = f"https://huggingface.co/{hf_repo_id}"
                subprocess.run(
                    ["git", "remote", "add", "origin", repo_url],
                    cwd=str(checkpoint_path),
                    check=False
                )
                
                # Configure git
                subprocess.run(
                    ["git", "config", "user.name", "training-bot"],
                    cwd=str(checkpoint_path),
                    check=False
                )
                subprocess.run(
                    ["git", "config", "user.email", "training@huggingface.co"],
                    cwd=str(checkpoint_path),
                    check=False
                )
                
                print(f"✓ Initialized new repository")
                print(f"✓ Checkpoint directory created: {checkpoint_path}")
                return True
        
        # Case 2: Directory exists but not a git repo
        elif not (checkpoint_path / ".git").exists():
            print(f"\n⚠️ Checkpoint directory exists but is not a git repository")
            print(f"Initializing as git repository...")
            
            # Initialize git
            subprocess.run(["git", "init"], cwd=str(checkpoint_path), check=True)
            
            # Add HF remote
            repo_url = f"https://huggingface.co/{hf_repo_id}"
            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=str(checkpoint_path),
                check=False
            )
            
            # Configure git
            subprocess.run(
                ["git", "config", "user.name", "training-bot"],
                cwd=str(checkpoint_path),
                check=False
            )
            subprocess.run(
                ["git", "config", "user.email", "training@huggingface.co"],
                cwd=str(checkpoint_path),
                check=False
            )
            
            # Try to pull from HF
            print(f"Attempting to pull from HF Hub...")
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(checkpoint_path),
                capture_output=True
            )
            
            if result.returncode == 0:
                print(f"✓ Pulled latest changes from HF Hub")
            else:
                print(f"ℹ️ Could not pull (repository may be empty)")
            
            print(f"✓ Repository initialized")
            return True
        
        # Case 3: Directory exists and is a git repo - Pull latest
        else:
            print(f"\n✓ Found existing git repository: {checkpoint_path}")
            print(f"Pulling latest changes from HF Hub...")
            
            result = subprocess.run(
                ["git", "pull"],
                cwd=str(checkpoint_path),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✓ Successfully pulled latest changes")
            else:
                print(f"⚠️ Could not pull changes: {result.stderr}")
                print(f"Continuing with local repository...")
            
            return True
    
    except Exception as e:
        print(f"\n❌ Error setting up checkpoint repository: {e}")
        print(f"⚠️ Continuing with local directory only...")
        
        # Create directory as fallback
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return False
    
    finally:
        print("="*70 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Connector-Aware Llama 3.2 Pretraining')
    
    parser.add_argument(
        '--resume-training',
        action='store_true',
        help='Resume training from the last checkpoint (pulls from HF if local not found)'
    )
    
    parser.add_argument(
        '--hf-repo-id',
        type=str,
        default=None,
        help='Hugging Face repository ID (e.g., username/model-name)'
    )
    
    parser.add_argument(
        '--hf-token',
        type=str,
        default=os.environ.get('HF_TOKEN'),
        help='Hugging Face API token (defaults to HF_TOKEN environment variable)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=2,
        help='Number of training epochs'
    )
    
    return parser.parse_args()


def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load configuration
        print("\n" + "="*70)
        print("CONNECTOR-AWARE LLAMA 3.2 PRETRAINING")
        print("="*70)
        
        cfg = Config()
        cfg.print_summary()
        
        # Print initial memory
        print_cuda_memory()
        
        # Setup checkpoint repository if HF repo is specified
        if args.hf_repo_id and args.hf_token:
            setup_checkpoint_repository(
                checkpoint_dir=cfg.checkpoint_dir,
                hf_repo_id=args.hf_repo_id,
                hf_token=args.hf_token
            )
        elif args.hf_repo_id:
            print("\n⚠️ WARNING: HF repository ID provided but no token found!")
            print("Set HF_TOKEN environment variable or pass --hf-token")
            print("Continuing with local checkpoints only...\n")
        
        # Initialize model wrapper
        model_wrapper = None
        trainer = None
        metadata = None
        start_file_idx = 0
        start_epoch = 1
        
        if args.resume_training:
            print("\n" + "="*70)
            print("RESUME TRAINING MODE")
            print("="*70)
            
            # Check if checkpoint directory exists
            checkpoint_path = Path(cfg.checkpoint_dir)
            
            if not checkpoint_path.exists():
                # Directory doesn't exist - try to pull from HF
                if args.hf_repo_id and args.hf_token:
                    print(f"\nCheckpoint directory not found locally")
                    print(f"Attempting to clone from HF Hub: {args.hf_repo_id}")
                    
                    setup_success = setup_checkpoint_repository(
                        checkpoint_dir=cfg.checkpoint_dir,
                        hf_repo_id=args.hf_repo_id,
                        hf_token=args.hf_token
                    )
                    
                    if not setup_success:
                        print("\n⚠️ Could not setup checkpoint repository")
                        print("Starting fresh training instead...")
                        args.resume_training = False
                else:
                    print(f"\n⚠️ Checkpoint directory not found: {checkpoint_path}")
                    print("⚠️ No HF repository specified to pull from")
                    print("Starting fresh training instead...")
                    args.resume_training = False
            
            # If still in resume mode, try to load checkpoint
            if args.resume_training:
                # Create temporary trainer to check for checkpoints
                temp_model = ConnectorAwareModel(cfg)
                temp_trainer = ConnectorTrainer(
                    temp_model, 
                    cfg, 
                    hf_repo_id=args.hf_repo_id,
                    hf_token=args.hf_token
                )
                
                # Try to load checkpoint
                metadata = temp_trainer.load_checkpoint()
                
                if metadata:
                    start_file_idx = metadata.get('files_processed', 0)
                    start_epoch = metadata.get('epoch', 1)
                    
                    print(f"\n✓ Resuming from checkpoint:")
                    print(f"  - Starting epoch: {start_epoch}")
                    print(f"  - Files already processed: {start_file_idx}")
                    print(f"  - Resuming from file: {start_file_idx + 1}")
                    
                    # Load model from checkpoint
                    print(f"\nLoading model from checkpoint: {temp_trainer.checkpoint_dir}")
                    model_wrapper = ConnectorAwareModel(cfg)
                    model_wrapper.model = type(model_wrapper.model).from_pretrained(
                        temp_trainer.checkpoint_dir,
                        torch_dtype=torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32,
                        device_map="auto"
                    )
                    model_wrapper.tokenizer = temp_model.tokenizer
                    
                    print(f"✓ Model loaded from checkpoint")
                else:
                    print("\n⚠️ No checkpoint found in repository")
                    print("Starting fresh training...")
                    args.resume_training = False
        
        # Initialize fresh model if not resuming or no checkpoint found
        if not args.resume_training or model_wrapper is None:
            print("\n" + "="*70)
            print("FRESH TRAINING MODE")
            print("="*70)
            print("Starting training from scratch\n")
            
            model_wrapper = ConnectorAwareModel(cfg)
        
        print(f"Model parameters: {model_wrapper.get_num_parameters():,}")
        print_cuda_memory()
        
        # Initialize trainer
        trainer = ConnectorTrainer(
            model_wrapper, 
            cfg,
            hf_repo_id=args.hf_repo_id,
            hf_token=args.hf_token
        )
        
        # Get connector tag IDs for streaming
        open_tags = cfg.get_special_tokens()[:-1]
        close_tag = cfg.closing_tag
        tokenizer = model_wrapper.tokenizer
        
        open_tag_ids = set(tokenizer.convert_tokens_to_ids(t) for t in open_tags)
        close_tag_id = tokenizer.convert_tokens_to_ids(close_tag)
        
        # Get training files
        train_files = sorted(glob.glob(f"{cfg.data_dir}/{cfg.train_pattern}"))
        print(f"\nFound {len(train_files)} training files")
        
        # Filter files if resuming
        if args.resume_training and start_file_idx > 0:
            train_files = train_files[start_file_idx:]
            print(f"Resuming from file index {start_file_idx}")
            print(f"Remaining files to process: {len(train_files)}")
        
        # Training loop
        for epoch in range(start_epoch, args.num_epochs + 1):
            try:
                print(f"\n{'='*70}")
                print(f"EPOCH {epoch}/{args.num_epochs}")
                print(f"{'='*70}\n")
                
                # Clear memory before each epoch
                print_cuda_memory()
                
                # Create batch streamer for this epoch
                batch_stream = token_batch_streamer(
                    files=train_files,
                    batch_size=cfg.batch_size,
                    open_tag_ids=open_tag_ids,
                    close_tag_id=close_tag_id,
                    boost=cfg.boost_factor
                )
                
                # Train epoch with checkpoint support
                avg_loss = trainer.train_epoch(
                    epoch_num=epoch,
                    batch_streamer=batch_stream,
                    files_to_process=len(train_files),
                    start_file_idx=start_file_idx if epoch == start_epoch else 0
                )
                
                # Reset start_file_idx after first epoch
                if epoch == start_epoch:
                    start_file_idx = 0
                
                # Clear memory after epoch
                clear_cuda_memory()
                
            except Exception as e:
                print(f"\n❌ Error during epoch {epoch}: {str(e)}")
                print("Attempting to clear memory...")
                clear_cuda_memory()
                raise
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print_cuda_memory()
        
        # Final cleanup
        clear_cuda_memory()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Clearing memory...")
        clear_cuda_memory()
        
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        print("Clearing memory...")
        clear_cuda_memory()
        raise


if __name__ == "__main__":
    main()