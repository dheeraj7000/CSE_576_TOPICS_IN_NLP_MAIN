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


def setup_checkpoint_repository(checkpoint_dir, hf_repo_id, hf_token):
    """Setup checkpoint directory as HF repository"""
    from huggingface_hub import HfApi, create_repo, Repository
    import subprocess
    
    checkpoint_path = Path(checkpoint_dir)
    
    print("\n" + "="*70)
    print("CHECKPOINT REPOSITORY SETUP")
    print("="*70)
    
    try:
        api = HfApi()
        try:
            create_repo(repo_id=hf_repo_id, token=hf_token, exist_ok=True, private=False, repo_type="model")
            print(f"✓ HF repository ready: {hf_repo_id}")
        except Exception as e:
            print(f"ℹ️ Repository may already exist: {e}")
        
        if not checkpoint_path.exists():
            print(f"\n📥 Cloning from HF Hub: {hf_repo_id}")
            try:
                Repository(local_dir=str(checkpoint_path), clone_from=hf_repo_id, token=hf_token,
                          git_user="training-bot", git_email="training@huggingface.co")
                print(f"✓ Successfully cloned")
                return True
            except Exception as e:
                print(f"ℹ️ Clone failed: {e}")
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init"], cwd=str(checkpoint_path), check=True)
                repo_url = f"https://huggingface.co/{hf_repo_id}"
                subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=str(checkpoint_path), check=False)
                print(f"✓ Initialized new repository")
                return True
        
        elif not (checkpoint_path / ".git").exists():
            print(f"\n⚠️ Initializing git repository")
            subprocess.run(["git", "init"], cwd=str(checkpoint_path), check=True)
            repo_url = f"https://huggingface.co/{hf_repo_id}"
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=str(checkpoint_path), check=False)
            print(f"✓ Repository initialized")
            return True
        else:
            print(f"\n✓ Found existing git repository")
            subprocess.run(["git", "pull"], cwd=str(checkpoint_path), capture_output=True)
            return True
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
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
        
        # Setup checkpoint repository
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
            
            checkpoint_path = Path(cfg.checkpoint_dir)
            
            if not checkpoint_path.exists() and args.hf_repo_id and args.hf_token:
                setup_checkpoint_repository(cfg.checkpoint_dir, args.hf_repo_id, args.hf_token)
            
            if args.resume_training:
                temp_model = ConnectorAwareModel(cfg)
                temp_trainer = ConnectorTrainer(temp_model, cfg, hf_repo_id=args.hf_repo_id, hf_token=args.hf_token)
                
                metadata = temp_trainer.load_checkpoint()
                
                if metadata:
                    start_file_idx = metadata.get('files_processed', 0)
                    start_epoch = metadata.get('epoch', 1)
                    
                    print(f"\n✓ Resuming from checkpoint:")
                    print(f"  - Starting epoch: {start_epoch}")
                    print(f"  - Files processed: {start_file_idx}")
                    
                    model_wrapper = ConnectorAwareModel(cfg)
                    model_wrapper.model = type(model_wrapper.model).from_pretrained(
                        temp_trainer.checkpoint_dir,
                        torch_dtype=torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32,
                        device_map="auto"
                    )
                    model_wrapper.tokenizer = temp_model.tokenizer
                else:
                    args.resume_training = False
        
        if not args.resume_training or model_wrapper is None:
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