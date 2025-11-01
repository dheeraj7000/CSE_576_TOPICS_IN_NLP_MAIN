#!/usr/bin/env python3
from utils.config import Config
from pretrain.model import Model
from pretrain.data_loader import PretrainingDataLoader
from huggingface_hub import login
import os


def loadModel(config):
    
    handler = Model(config)
    
    tokenizer = handler.load_tokenizer()
    
    special_tokens = config.get_special_tokens()
    print(f"Tokens to add: {special_tokens}")

    num_added = handler.extend_tokenizer(special_tokens)
    print(f"Added {num_added} tokens")
    print(f"New vocab size: {len(tokenizer):,}")
    
    handler.load_model()
    
    info = handler.get_model_info()
    print(f"\nTotal parameters: {info['total_parameters']:,}")
    print(f"Device: {info['device']}")
    
    handler.initialize_new_embeddings(config.connector_types)
    
    print("\nModel is ready for training!")
    
    return handler

def loadData(config):
    """
    Demonstrates loading the data using the new data loader class.
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_loader = PretrainingDataLoader(config)
    
    # Load the preprocessed data (split into train/test)
    # This loads from the './checkpoints' directory by default
    training_data_dict = data_loader.load_preprocessed_data_for_training(test_split_size=0.01)
    
    if training_data_dict:
        print(f"\n✓ Successfully loaded training data.")
        print(f"  Training samples: {len(training_data_dict['train']):,}")
        print(f"  Test samples: {len(training_data_dict['test']):,}")
        # print("\nSample (from train split):")
        # print(training_data_dict['train'][0])
    else:
        print("\n✗ Failed to load training data. Please run preprocessing first.")
    
    eval_dataset = data_loader.load_evaluation_data("logiqa")
    
    if eval_dataset:
        print(f"\n✓ Successfully loaded LogiQA evaluation data.")
        print(f"  LogiQA samples: {len(eval_dataset):,}")
        # print("\nSample (from LogiQA):")
        # print(eval_dataset[0])
    else:
        print("\n✗ Failed to load LogiQA evaluation data.")
        
    print("="*80)
    return training_data_dict, eval_dataset


def main():
    token = os.environ.get('HF_TOKEN')
    login(token = token)
    config = Config()

    modelHandel = loadModel(config)
    
    training_data, eval_data = loadData(config)
    
    print("\nNext step: Initialize and run the Trainer.")


if __name__ == "__main__":
    main()
