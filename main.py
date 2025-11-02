#!/usr/bin/env python3
from utils.config import Config
from pretrain.model import Model
from huggingface_hub import login
import os

def loadModel(config: Config):
    
    model_handler = Model(config)
    
    tokenizer = model_handler.load_tokenizer()
    
    # Add special tokens including connector markup tokens
    special_tokens = config.get_special_tokens() + ["<connector>", "</connector>"]
    print(f"Tokens to add: {special_tokens}")
    num_added = model_handler.extend_tokenizer(special_tokens)
    print(f"Added {num_added} tokens")
    print(f"New vocab size: {len(tokenizer):,}")
    
    # Load model
    model_handler.load_model()
    
    def initialize_with_logging(connector_taxonomy):
        embedding_layer = model_handler.model.get_input_embeddings()
        current_vocab_size = len(model_handler.tokenizer)
        fallback_tokens = []

        with torch.no_grad():
            existing_embeddings = embedding_layer.weight[:model_handler.original_vocab_size].clone()
            
            for token_id in range(model_handler.original_vocab_size, current_vocab_size):
                token = model_handler.tokenizer.convert_ids_to_tokens(token_id)
                initialized = False
                
                for conn_type, example_words in connector_taxonomy.items():
                    if conn_type.lower() in str(token).lower():
                        similar_ids = []
                        for word in example_words[:5]:
                            word_tokens = model_handler.tokenizer.tokenize(word)
                            if word_tokens:
                                word_id = model_handler.tokenizer.convert_tokens_to_ids(word_tokens)
                                if isinstance(word_id, list):
                                    word_id = [i for i in word_id if i < model_handler.original_vocab_size]
                                    similar_ids.extend(word_id)
                                elif word_id < model_handler.original_vocab_size:
                                    similar_ids.append(word_id)
                        
                        if similar_ids:
                            avg_embedding = existing_embeddings[similar_ids].mean(dim=0)
                            embedding_layer.weight.data[token_id] = avg_embedding
                            initialized = True
                            break
                
                if not initialized:
                    embedding_layer.weight.data[token_id] = existing_embeddings.mean(dim=0)
                    fallback_tokens.append(token)
        
        if fallback_tokens:
            print(f"Fallback embeddings used for tokens: {fallback_tokens}")

    model_handler.initialize_new_embeddings = initialize_with_logging

    model_handler.initialize_new_embeddings(config.connector_types)
    
    # Show model info
    info = model_handler.get_model_info()
    print(f"\nTotal parameters: {info['total_parameters']:,}")
    print(f"Device: {info['device']}")
    
    pretrain_manager = ConnectorPretrainingManager(config, model_handler)
    
    print("\n✓ Model ready for connector-aware pretraining!")
    
    return model_handler, pretrain_manager


def main():
    # HuggingFace login
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
    
    config = Config()
    
    model_handler, pretrain_manager = loadModel(config)
    
    print("\n✓ Main pipeline setup complete. Ready to prepare datasets and train.")


if __name__ == "__main__":
    main()
