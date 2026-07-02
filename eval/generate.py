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
    ap.add_argument("--output_file", default=None, help="optional path to save generated samples")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)

    # --- load checkpoint + rebuild the exact model -------------------------
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"loaded checkpoint from iter {ckpt['iter']} (val loss {ckpt['best_val_loss']:.4f})")

    # --- load the codec from the dataset meta ------------------------------
    data_dir = ckpt["config"]["data_dir"]
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta.get("tokenizer_type", "char") == "bpe":
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit("tokenizers is required for BPE checkpoints; run `pip install -r requirements.txt`.") from exc

        tokenizer_path = os.path.join(data_dir, meta["tokenizer_file"])
        tokenizer = Tokenizer.from_file(tokenizer_path)
        encode = lambda s: tokenizer.encode(s).ids
        decode = lambda ids: tokenizer.decode(ids)
    else:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[i] for i in ids)

        missing = sorted(set(args.start) - set(stoi))
        if missing:
            raise ValueError(f"prompt contains characters outside the training vocabulary: {missing!r}")

    start_ids = encode(args.start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # --- sample ------------------------------------------------------------
    chunks = [
        f"checkpoint: {args.out_dir}",
        f"iter: {ckpt['iter']}",
        f"best_val_loss: {ckpt['best_val_loss']:.4f}",
        f"prompt: {args.start!r}",
        "",
    ]
    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        sample = decode(y[0].tolist())
        print(f"\n----- sample {i + 1} -----")
        print(sample)
        chunks.extend([f"----- sample {i + 1} -----", sample, ""])

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
        print(f"\nwrote samples -> {args.output_file}")


if __name__ == "__main__":
    main()
