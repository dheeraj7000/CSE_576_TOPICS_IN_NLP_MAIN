# pretrain/main.py
import glob
from tqdm import tqdm
from utils.config import Config
from pretrain.data_loader import token_batch_streamer
from transformers import AutoTokenizer

def run_epoch(cfg: Config, epoch: int):
    print(f"\n==== Starting Epoch {epoch} ====")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    open_tags = cfg.get_special_tokens()[:-1]
    close_tag = cfg.closing_tag
    open_tag_ids = set(tokenizer.convert_tokens_to_ids(t) for t in open_tags)
    close_tag_id = tokenizer.convert_tokens_to_ids(close_tag)
    batch_files = sorted(glob.glob(f"{cfg.data_dir}/train_chunk_*.parquet"))

    print(f"Total train files: {len(batch_files)}")
    print(f"Processing in batches of {cfg.batch_size}")

    # Count total tokens first if you want an accurate total number of batches (optional, may be slow on huge data)
    # Otherwise, just keep tqdm as an open-ended progress bar
    batches = token_batch_streamer(
        files=batch_files,
        batch_size=cfg.batch_size,
        open_tag_ids=open_tag_ids,
        close_tag_id=close_tag_id,
        boost=cfg.boost_factor
    )
    batch_count = 0
    for tokens, amps, sources in tqdm(batches, desc=f"Epoch {epoch} Batches", unit="batch"):
        batch_count += 1
        # Insert model training step here if needed
        # Example print for first few batches:
        if batch_count <= cfg.max_batches_to_print:
            print(f"--- Batch {batch_count} ---\nDecoded: {tokenizer.decode(tokens, skip_special_tokens=False)}\nAmps: {amps}")
    print(f"Epoch {epoch}: processed {batch_count} batches.")

if __name__ == "__main__":
    cfg = Config()
    cfg.print_summary()
    for epoch in range(1, 3):
        run_epoch(cfg, epoch)
