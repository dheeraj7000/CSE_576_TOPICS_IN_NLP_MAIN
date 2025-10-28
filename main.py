#!/usr/bin/env python3
from utils.config import Config
from pretrain.model import Model
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

def main():
    token = os.environ.get('HF_TOKEN')
    login(token = token)
    config = Config()

    modelHandel = loadModel(config)



if __name__ == "__main__":
    main()
