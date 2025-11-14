import glob
import pandas as pd
from transformers import AutoTokenizer
from utils.config import Config
import numpy as np

# --- LOAD CONFIG & TOKENIZER ---
cfg = Config()

# --- USER CONFIGURABLE ---
BATCH_SIZE = cfg.batch_size
FILES_PER_CHUNK = cfg.files_per_chunk
DATA_DIR = cfg.data_dir
MAX_BATCHES_TO_PRINT = cfg.max_batches_to_print
TOKENIZER_PATH = cfg.tokenizer_path

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Dynamically get tag tokens from config
OPEN_TAGS = cfg.get_special_tokens()[:-1]
CLOSE_TAG = cfg.closing_tag

open_tag_ids = set(tokenizer.convert_tokens_to_ids(t) for t in OPEN_TAGS)
close_tag_id = tokenizer.convert_tokens_to_ids(CLOSE_TAG)

BOOST = cfg.boost_factor

train_files = sorted(glob.glob(f"{DATA_DIR}/train_chunk_*.parquet"))


# --- STREAM ONE CHUNK AT A TIME ---
def stream_tokens_from_files(files, file_callback=None):
    for fidx, fname in enumerate(files):
        # Call callback when loading new file
        if file_callback:
            file_callback(fidx, fname)
        
        df = pd.read_parquet(fname)
        
        for sidx, row in enumerate(df.itertuples(index=False)):
            input_ids = list(row.input_ids)
            for p, tid in enumerate(input_ids):
                yield tid, fidx, sidx, p, fname


# --- BATCHER ---
def token_batch_streamer(files, batch_size, open_tag_ids, close_tag_id, boost, max_batches=None):
    file_pointer = 0
    total_files = len(files)
    printed_batches = 0
    inside_connector = False
    
    batch_tokens = []
    batch_amps = []
    batch_sources = []
    
    current_file_name = None
    
    def on_file_load(fidx, fname):
        """Callback when a new file is loaded"""
        nonlocal current_file_name
        current_file_name = fname
        print(f"\n📂 Loading file {fidx + file_pointer}/{total_files}: {fname}")
    
    while file_pointer < total_files and (max_batches is None or printed_batches < max_batches):
        chunk_files = files[file_pointer:file_pointer+FILES_PER_CHUNK]
        
        if not chunk_files:
            break
        
        print(f"\n{'='*70}")
        print(f"Processing file chunk ({file_pointer} to {file_pointer+len(chunk_files)-1})")
        print(f"{'='*70}")
        
        token_stream = stream_tokens_from_files(chunk_files, file_callback=on_file_load)
        
        for tid, fidx, sidx, pos, fname in token_stream:
            # Amplification logic
            if tid in open_tag_ids:
                inside_connector = True
                amp = 1.0
            elif tid == close_tag_id:
                amp = 1.0
                inside_connector = False
            else:
                amp = boost if inside_connector else 1.0
            
            batch_tokens.append(int(tid))
            batch_amps.append(float(amp))
            batch_sources.append((fidx+file_pointer, sidx, pos, fname))
            
            if len(batch_tokens) == batch_size:
                yield batch_tokens[:], batch_amps[:], batch_sources[:]
                batch_tokens.clear()
                batch_amps.clear()
                batch_sources.clear()
                
                printed_batches += 1
                
                if max_batches is not None and printed_batches >= max_batches:
                    return
        
        file_pointer += FILES_PER_CHUNK
    
    # Yield remaining tokens
    if batch_tokens and (max_batches is None or printed_batches < max_batches):
        yield batch_tokens, batch_amps, batch_sources


# --- MAIN: TEST BATCH STREAMING OUTPUT ---
def tokens_to_str(tokens):
    try:
        return tokenizer.decode(tokens, skip_special_tokens=False)
    except Exception:
        return str(tokens)


if __name__ == "__main__":
    print(f"Found {len(train_files)} train files. Will process in chunks of {FILES_PER_CHUNK}.\n")
    print("================ STREAMING AMPLIFICATION BATCH TEST ================\n")
    print(f"Config model: {cfg.model_name} | Batch size: {BATCH_SIZE} | Files/chunk: {FILES_PER_CHUNK}\n")
    print(f"First {MAX_BATCHES_TO_PRINT} batches:\n")
    
    for i, (tokens, amps, sources) in enumerate(token_batch_streamer(
        train_files, BATCH_SIZE, open_tag_ids, close_tag_id, BOOST, max_batches=MAX_BATCHES_TO_PRINT
    )):
        print(f"\n================= BATCH {i+1} =================")
        print("Tokens:")
        print(tokens)
        print("Amps:")
        print(amps)
        print("Decoded:")
        print(tokens_to_str(tokens))
    
    print("\n===== DONE =====\n")