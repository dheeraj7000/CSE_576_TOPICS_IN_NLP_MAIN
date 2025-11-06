#!/usr/bin/env python3
"""
main.py - UPDATED

Main entry point for connector-aware pretraining.
Automatically trains for 1 epoch and saves the model.
"""

import os
import torch
from huggingface_hub import login

# Import project modules
from utils.config import Config
from pretrain.model import Model
from pretrain.data_loader import PretrainingDataLoader, ConnectorDataCollatorWithMaskCreation
from pretrain.trainer import ConnectorPretrainingManager


def setup_huggingface():
    """Login to HuggingFace if token available"""
    token = os.environ.get('HF_TOKEN')
    if token:
        print("Logging into HuggingFace...")
        login(token=token)
        print("✓ Logged in")
    else:
        print("⚠ No HF_TOKEN found. Some models may be inaccessible.")


def load_model(config: Config):
    """
    Load and prepare model with connector tokens.
    
    Args:
        config: Configuration object
        
    Returns:
        model_handler: Initialized Model instance
    """
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    model_handler = Model(config)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = model_handler.load_tokenizer()
    print(f"✓ Original vocab size: {len(tokenizer):,}")
    
    # Add special tokens
    print("\n[2/4] Adding special tokens...")
    special_tokens = config.get_special_tokens()
    print(f"Tokens to add: {special_tokens}")
    num_added = model_handler.extend_tokenizer(special_tokens)
    print(f"✓ Added {num_added} tokens")
    print(f"✓ New vocab size: {len(tokenizer):,}")
    
    # Load model
    print("\n[3/4] Loading model...")
    model_handler.load_model()
    
    # Initialize embeddings
    print("\n[4/4] Initializing new embeddings...")
    initialize_embeddings_with_logging(model_handler, config)
    
    # Show model info
    info = model_handler.get_model_info()
    print(f"\n✓ Total parameters: {info['total_parameters']:,}")
    print(f"✓ Trainable parameters: {info['trainable_parameters']:,}")
    print(f"✓ Device: {info['device']}")
    print(f"✓ Dtype: {info['dtype']}")
    if info['connector_boosting']:
        print(f"✓ Connector boosting: {info['boost_factor']}x")
    
    print("\n" + "=" * 70)
    print("✓ Model loaded successfully")
    print("=" * 70)
    
    return model_handler


def initialize_embeddings_with_logging(model_handler, config):
    """
    Initialize new token embeddings using semantic averaging.
    
    Args:
        model_handler: Model instance
        config: Configuration object
    """
    embedding_layer = model_handler.model.get_input_embeddings()
    current_vocab_size = len(model_handler.tokenizer)
    fallback_tokens = []
    
    with torch.no_grad():
        existing_embeddings = embedding_layer.weight[:model_handler.original_vocab_size].clone()
        
        for token_id in range(model_handler.original_vocab_size, current_vocab_size):
            token = model_handler.tokenizer.convert_ids_to_tokens(token_id)
            initialized = False
            
            # Try to initialize from similar words
            for conn_type, example_words in config.connector_types.items():
                if conn_type.lower() in str(token).lower():
                    similar_ids = []
                    
                    for word in example_words[:5]:  # Use first 5 examples
                        word_tokens = model_handler.tokenizer.tokenize(word)
                        if word_tokens:
                            word_id = model_handler.tokenizer.convert_tokens_to_ids(word_tokens[0])
                            if word_id < model_handler.original_vocab_size:
                                similar_ids.append(word_id)
                    
                    if similar_ids:
                        avg_embedding = existing_embeddings[similar_ids].mean(dim=0)
                        embedding_layer.weight.data[token_id] = avg_embedding
                        print(f"   ✓ '{token}' initialized from {len(similar_ids)} similar words")
                        initialized = True
                        break
            
            # Fallback: use mean of all embeddings
            if not initialized:
                embedding_layer.weight.data[token_id] = existing_embeddings.mean(dim=0)
                fallback_tokens.append(token)
    
    if fallback_tokens:
        print(f"   ℹ Fallback embeddings used for {len(fallback_tokens)} tokens")


def load_data(config: Config):
    """
    Load preprocessed training data.
    
    Args:
        config: Configuration object
        
    Returns:
        dataset_dict: Dictionary with 'train' and 'test' datasets
    """
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    data_loader = PretrainingDataLoader(config)
    
    dataset_dict = data_loader.load_preprocessed_data_for_training(
        test_split_size=0.05,
        seed=42
    )
    
    if dataset_dict is None:
        print("\n❌ Failed to load data!")
        print("💡 Please run preprocessing first:")
        print("   python preprocess.py")
        return None
    
    print("\n" + "=" * 70)
    print("✓ Data loaded successfully")
    print("=" * 70)
    
    return dataset_dict


def run_training(config, model_handler, dataset_dict):
    """
    Run training for 1 epoch and save the model.
    
    Args:
        config: Configuration object
        model_handler: Model instance
        dataset_dict: Dictionary with 'train' and 'test' datasets
    """
    print("\n" + "=" * 70)
    print("SETTING UP TRAINING")
    print("=" * 70)
    
    # Create training manager
    print("\n[1/3] Creating training manager...")
    pretrain_manager = ConnectorPretrainingManager(
        config=config,
        model_handler=model_handler,
        use_new_collator=True  # Use new collator from data_loader.py
    )
    print("✓ Training manager created")
    
    # Prepare trainer
    print("\n[2/3] Preparing trainer...")
    pretrain_manager.prepare_trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['test'],
        output_dir="./output/connector_model_1epoch",
        
        # Connector boosting settings
        boost_factor=1.1,              # 10% boost for connectors
        use_amplification=True,         # Additional amplification
        amplification_strength=1.2,     # 20% additional boost
        
        # Training settings (1 EPOCH)
        num_epochs=1,                   # Train for 1 epoch only
        batch_size=2,
        learning_rate=5e-6
    )
    print("✓ Trainer prepared")
    
    # Start training
    print("\n[3/3] Starting training (1 epoch)...")
    print("\n" + "=" * 70)
    print("TRAINING IN PROGRESS")
    print("=" * 70)
    
    pretrain_manager.train()
    
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    
    return pretrain_manager


def save_model(pretrain_manager):
    """
    Save the trained model.
    
    Args:
        pretrain_manager: Training manager instance
    """
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    output_path = "./output/connector_model_1epoch/final"
    pretrain_manager.save_model(output_path)
    
    print("\n✓ Model saved to:", output_path)
    print("=" * 70)


def main():
    """Main entry point - Runs full training pipeline"""
    print("\n" + "=" * 70)
    print("CONNECTOR-AWARE PRETRAINING PIPELINE")
    print("Training for 1 epoch with automatic save")
    print("=" * 70)
    
    try:
        # Step 1: Setup HuggingFace
        setup_huggingface()
        
        # Step 2: Load configuration
        print("\n[Step 1/5] Loading configuration...")
        config = Config()
        print(f"✓ Model: {config.model_name}")
        print(f"✓ Connector types: {len(config.connector_types)}")
        print(f"✓ Boost factor: {config.boost_factor}x")
        
        # Step 3: Load model
        print("\n[Step 2/5] Loading model...")
        model_handler = load_model(config)
        
        # Step 4: Load data
        print("\n[Step 3/5] Loading data...")
        dataset_dict = load_data(config)
        
        if dataset_dict is None:
            print("\n❌ Pipeline failed: No data loaded")
            print("\n💡 To continue:")
            print("   1. Run preprocessing: python preprocess.py")
            print("   2. Then run this script again: python main.py")
            return
        
        # Step 5: Train model
        print("\n[Step 4/5] Training model (1 epoch)...")
        pretrain_manager = run_training(config, model_handler, dataset_dict)
        
        # Step 6: Save model
        print("\n[Step 5/5] Saving model...")
        save_model(pretrain_manager)
        
        # Success!
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70)
        print("\n📊 Training Summary:")
        print(f"   • Model: {config.model_name}")
        print(f"   • Training samples: {len(dataset_dict['train']):,}")
        print(f"   • Eval samples: {len(dataset_dict['test']):,}")
        print(f"   • Epochs: 1")
        print(f"   • Boost factor: {config.boost_factor}x")
        print(f"   • Output: ./output/connector_model_1epoch/final")
        print("\n🎉 Model trained and saved successfully!")
        print("\n💡 Next steps:")
        print("   • Load model: AutoModelForCausalLM.from_pretrained('./output/connector_model_1epoch/final')")
        print("   • Continue training: Increase num_epochs in run_training()")
        print("   • Evaluate: Run evaluation script on saved model")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("💡 Progress may be saved in checkpoints")
        
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check error above and ensure:")
        print("   • Preprocessing is complete")
        print("   • Model is accessible (check HF_TOKEN)")
        print("   • Sufficient GPU memory available")


if __name__ == "__main__":
    main()
