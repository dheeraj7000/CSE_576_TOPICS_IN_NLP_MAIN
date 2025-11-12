#!/usr/bin/env python3
"""
save_tokenizer.py

Utility to save the extended tokenizer (with connector tokens) to disk.
This allows you to reuse the same tokenizer consistently.
"""

import sys
from pathlib import Path


def save_extended_tokenizer(output_dir: str = "./tokenizer_extended"):
    """
    Load tokenizer from config, add connector tokens, and save to disk.
    
    Args:
        output_dir: Directory to save tokenizer
    """
    print("\n" + "="*70)
    print("TOKENIZER SAVER")
    print("="*70)
    
    try:
        # Load config
        print("\n[1/4] Loading config...")
        from utils.config import Config
        config = Config()
        print(f"✓ Model: {config.model_name}")
        
        # Load base tokenizer
        print("\n[2/4] Loading base tokenizer...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        original_vocab_size = len(tokenizer)
        print(f"✓ Original vocab size: {original_vocab_size:,}")
        
        # Add connector tokens
        print("\n[3/4] Adding connector tokens...")
        special_tokens = config.get_special_tokens()
        print(f"Tokens to add:")
        for token in special_tokens:
            print(f"  • {token}")
        
        num_added = tokenizer.add_tokens(special_tokens)
        new_vocab_size = len(tokenizer)
        
        print(f"\n✓ Added {num_added} tokens")
        print(f"✓ New vocab size: {new_vocab_size:,}")
        
        # Save to disk
        print(f"\n[4/4] Saving extended tokenizer...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Tokenizer saved to: {output_dir}")
        
        # Test loading
        print(f"\n[Test] Verifying saved tokenizer...")
        test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print(f"✓ Loaded from disk")
        print(f"  Vocab size: {len(test_tokenizer):,}")
        
        # Test encoding/decoding
        test_text = "This is a test <connector type=\"causal\"> because </connector> example."
        test_ids = test_tokenizer.encode(test_text)
        decoded = test_tokenizer.decode(test_ids)
        print(f"\n[Test] Encoding/Decoding:")
        print(f"  Input:   {test_text}")
        print(f"  Encoded: {len(test_ids)} tokens")
        print(f"  Decoded: {decoded}")
        
        print("\n" + "="*70)
        print("✓ SUCCESS!")
        print("="*70)
        print(f"\nTokenizer saved to: {output_dir}")
        print(f"\nTo use this tokenizer:")
        print(f"  from transformers import AutoTokenizer")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Save extended tokenizer with connector tokens"
    )
    parser.add_argument(
        '--output', '-o',
        default='./tokenizer_extended',
        help='Output directory (default: ./tokenizer_extended)'
    )
    
    args = parser.parse_args()
    
    success = save_extended_tokenizer(args.output)
    sys.exit(0 if success else 1)