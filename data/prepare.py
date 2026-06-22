"""
Prepare the tiny-shakespeare dataset for CHARACTER-LEVEL language modeling.

This is the "step 0" data pipeline for the gpt2-to-2026 upgrade chain. We start
with a character-level tokenizer (one token == one character) on purpose: it is
the simplest possible tokenizer, so the first baseline has zero moving parts
beyond the model itself. A real BPE tokenizer arrives later as its own upgrade.

Outputs (under data/shakespeare_char/):
  - train.bin / val.bin : token ids as a flat uint16 array (np.memmap-friendly)
  - meta.pkl            : {vocab_size, stoi, itos} to encode/decode at sample time

Run:
  .venv/bin/python data/prepare.py
"""

import os
import pickle
import urllib.request

import numpy as np

# Karpathy's mirror of the tiny-shakespeare corpus (~1 MB of plain text).
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "shakespeare_char")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    input_path = os.path.join(OUT_DIR, "input.txt")

    # 1. Download the raw text once.
    if not os.path.exists(input_path):
        print(f"downloading tiny-shakespeare -> {input_path}")
        urllib.request.urlretrieve(DATA_URL, input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # 2. Build the character-level vocabulary (sorted -> deterministic ids).
    chars = sorted(set(data))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}  # string  -> int
    itos = {i: ch for i, ch in enumerate(chars)}  # int     -> string
    print(f"vocab size: {vocab_size}")

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]

    # 3. Train / val split (90 / 10).
    n = len(data)
    train_data, val_data = data[: int(n * 0.9)], data[int(n * 0.9) :]
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val   has {len(val_ids):,} tokens")

    # 4. Persist token ids as raw uint16 so train.py can np.memmap them cheaply.
    train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
    val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

    # 5. Save the codec + vocab_size (needed at sampling time and by the model).
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote train.bin, val.bin, meta.pkl -> {OUT_DIR}")


if __name__ == "__main__":
    main()
