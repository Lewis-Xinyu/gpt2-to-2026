"""
Prepare a tiny-shakespeare BPE dataset for Step 1.

This keeps the raw text and train/val split identical to data/prepare.py, but
replaces the character vocabulary with a learned byte-level BPE tokenizer.

Outputs (under data/shakespeare_bpe/):
  - train.bin / val.bin : token ids as a flat integer array
  - meta.pkl            : tokenizer metadata for train.py and generate.py
  - tokenizer.json      : serialized Hugging Face tokenizers model

Run:
  python data/prepare_bpe.py --vocab_size 8000
"""

from __future__ import annotations

import argparse
import os
import pickle
import urllib.request

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "shakespeare_bpe")
SPECIAL_TOKENS = ["<|endoftext|>"]


def ensure_raw_text(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    input_path = os.path.join(out_dir, "input.txt")
    if not os.path.exists(input_path):
        print(f"downloading tiny-shakespeare -> {input_path}")
        urllib.request.urlretrieve(DATA_URL, input_path)
    return input_path


def build_tokenizer(input_path: str, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train([input_path], trainer)
    return tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    input_path = ensure_raw_text(args.out_dir)
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    tokenizer = build_tokenizer(input_path, args.vocab_size)
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"bpe vocab size: {vocab_size}")

    n = len(data)
    train_text, val_text = data[: int(n * 0.9)], data[int(n * 0.9) :]
    train_ids = tokenizer.encode(train_text).ids
    val_ids = tokenizer.encode(val_text).ids
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    np.array(train_ids, dtype=dtype).tofile(os.path.join(args.out_dir, "train.bin"))
    np.array(val_ids, dtype=dtype).tofile(os.path.join(args.out_dir, "val.bin"))
    print(f"train has {len(train_ids):,} tokens")
    print(f"val   has {len(val_ids):,} tokens")

    meta = {
        "tokenizer_type": "bpe",
        "tokenizer_file": "tokenizer.json",
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote train.bin, val.bin, meta.pkl, tokenizer.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
