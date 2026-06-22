"""
generate.py  —  sample text from a trained checkpoint.

Run:
  .venv/bin/python eval/generate.py --out_dir out/baseline --start "ROMEO:"
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch

# make the repo root importable when run as `python eval/generate.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.baseline_gpt import GPT, GPTConfig  # noqa: E402


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out/baseline")
    ap.add_argument("--start", default="\n", help="prompt text to continue")
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)

    # --- load checkpoint + rebuild the exact model -------------------------
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"loaded checkpoint from iter {ckpt['iter']} (val loss {ckpt['best_val_loss']:.4f})")

    # --- load the char codec (stoi/itos) from the dataset meta -------------
    data_dir = ckpt["config"]["data_dir"]
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    start_ids = encode(args.start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # --- sample ------------------------------------------------------------
    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"\n----- sample {i + 1} -----")
        print(decode(y[0].tolist()))


if __name__ == "__main__":
    main()
