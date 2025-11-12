#!/usr/bin/env python3
"""
pretrain/main.py

Entry point for connector-aware Llama 3.2 pretraining.
Integrates config, data loader, model, and trainer.
"""

import glob
import torch
from utils.config import Config
from pretrain.data_loader import token_batch_streamer
from pretrain.model import ConnectorAwareModel
from pretrain.trainer import ConnectorTrainer
from utils.clear_memory import clear_cuda_memory, print_cuda_memory

def main():
    try:
        # Load configuration
        print("\n" + "="*70)
        print("CONNECTOR-AWARE LLAMA 3.2 PRETRAINING")
        print("="*70)
        
        cfg = Config()
        cfg.print_summary()
        
        # Print initial memory
        print_cuda_memory()
        
        # Initialize model
        model_wrapper = ConnectorAwareModel(cfg)
        print(f"Model parameters: {model_wrapper.get_num_parameters():,}")
        print_cuda_memory()
        
        # Initialize trainer
        trainer = ConnectorTrainer(model_wrapper, cfg)
        
        # Get connector tag IDs for streaming
        open_tags = cfg.get_special_tokens()[:-1]
        close_tag = cfg.closing_tag
        tokenizer = model_wrapper.tokenizer
        
        open_tag_ids = set(tokenizer.convert_tokens_to_ids(t) for t in open_tags)
        close_tag_id = tokenizer.convert_tokens_to_ids(close_tag)
        
        # Get training files
        train_files = sorted(glob.glob(f"{cfg.data_dir}/{cfg.train_pattern}"))
        print(f"\nFound {len(train_files)} training files")
        
        # Training loop
        num_epochs = 2
        
        for epoch in range(1, num_epochs + 1):
            try:
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
                
                # Train epoch
                avg_loss = trainer.train_epoch(epoch, batch_stream)
                
                # Save checkpoint after each epoch
                checkpoint_path = f"./checkpoints/epoch_{epoch}"
                trainer.save_checkpoint(checkpoint_path)
                
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
        print("\n\n⚠️  Training interrupted by user")
        print("Clearing memory...")
        clear_cuda_memory()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        print("Clearing memory...")
        clear_cuda_memory()
        raise

if __name__ == "__main__":
    main()
