"""
train.py  —  the config-driven training harness for the upgrade chain.

One YAML config in, one trained checkpoint out. Every ablation reuses this exact
loop so that comparisons are fair: same data sampling, same seed, same schedule,
same budget. The only thing that changes between experiments is the config (and,
later, which model component is swapped in).

Run:
  .venv/bin/python train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import yaml

from model.baseline_gpt import GPT, GPTConfig


# --------------------------------------------------------------------------- #
# setup helpers
# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_lr(it: int, t: dict) -> float:
    """Linear warmup then cosine decay down to min_lr (the GPT-2/nanoGPT schedule)."""
    if it < t["warmup_iters"]:
        return t["learning_rate"] * (it + 1) / (t["warmup_iters"] + 1)
    if it > t["lr_decay_iters"]:
        return t["min_lr"]
    ratio = (it - t["warmup_iters"]) / (t["lr_decay_iters"] - t["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return t["min_lr"] + coeff * (t["learning_rate"] - t["min_lr"])


def build_gpt_config(model_cfg: dict) -> GPTConfig:
    """Merge YAML model overrides with GPTConfig defaults.

    This keeps old experiment configs valid as new switchable components are
    added to GPTConfig, while still catching misspelled config keys.
    """
    defaults = asdict(GPTConfig())
    unknown = sorted(set(model_cfg) - set(defaults))
    if unknown:
        raise ValueError(f"unknown model config keys: {unknown}")
    defaults.update(model_cfg)
    return GPTConfig(**defaults)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    m_cfg, t = cfg["model"], cfg["train"]
    device = pick_device(t.get("device", "auto"))
    device_type = "cuda" if device.startswith("cuda") else device
    log_interval = t.get("log_interval", 50)

    torch.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))

    out_dir = t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    # --- data (memory-mapped token streams) --------------------------------
    data_dir = cfg["data_dir"]
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    token_dtype = np.dtype(meta.get("dtype", "uint16"))
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=token_dtype, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=token_dtype, mode="r")
    block_size = m_cfg["block_size"]
    batch_size = t["batch_size"]

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    # --- model -------------------------------------------------------------
    m_cfg["vocab_size"] = meta["vocab_size"]  # fill from data
    gptconf = build_gpt_config(m_cfg)
    resolved_cfg = dict(cfg)
    resolved_cfg["model"] = asdict(gptconf)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)
    model = GPT(gptconf).to(device)
    print(f"model params (non-embedding): {model.get_num_params():,}")

    optimizer = model.configure_optimizers(
        t["weight_decay"], t["learning_rate"], (t["beta1"], t["beta2"]), device_type
    )

    # autocast only pays off on CUDA; keep MPS/CPU in fp32 for a stable baseline.
    use_amp = device_type == "cuda" and t.get("dtype", "auto") != "fp32"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    @torch.no_grad()
    def estimate_loss() -> dict:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(t["eval_iters"])
            for k in range(t["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # --- csv log -----------------------------------------------------------
    log_path = os.path.join(out_dir, "log.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["iter", "split", "loss", "lr", "dt_ms", "tok_per_sec"])

    # --- training loop -----------------------------------------------------
    best_val = float("inf")
    last_eval = None
    tokens_per_iter = batch_size * block_size
    t0 = time.time()
    model.train()

    for it in range(t["max_iters"] + 1):
        lr = get_lr(it, t)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # periodic evaluation + checkpoint
        if it % t["eval_interval"] == 0:
            losses = estimate_loss()
            last_eval = {"iter": it, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr}
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}")
            logger.writerow([it, "eval_train", f"{losses['train']:.4f}", f"{lr:.2e}", "", ""])
            logger.writerow([it, "eval_val", f"{losses['val']:.4f}", f"{lr:.2e}", "", ""])
            log_file.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_args": asdict(gptconf),
                        "iter": it,
                        "best_val_loss": best_val,
                        "config": resolved_cfg,
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )

        if it == t["max_iters"]:
            break

        # one optimization step
        X, Y = get_batch("train")
        with ctx:
            _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if t["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        optimizer.step()

        # throughput logging (the numbers that power the ablation tables later)
        if it % log_interval == 0:
            if device_type == "mps":
                torch.mps.synchronize()
            elif device_type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            dt_per_iter = dt / log_interval if it > 0 else dt
            tok_s = tokens_per_iter / dt_per_iter if dt_per_iter > 0 else 0
            print(f"  step {it:5d} | loss {loss.item():.4f} | {dt_per_iter*1e3:6.1f} ms/iter | {tok_s:8.0f} tok/s")
            logger.writerow([it, "train", f"{loss.item():.4f}", f"{lr:.2e}", f"{dt_per_iter*1e3:.1f}", f"{tok_s:.0f}"])
            t0 = time.time()

    log_file.close()
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "name": cfg.get("name"),
                "seed": cfg.get("seed"),
                "device": device,
                "dtype": str(amp_dtype).replace("torch.", "") if use_amp else "fp32",
                "best_val_loss": best_val,
                "last_eval": last_eval,
                "checkpoint": os.path.join(out_dir, "ckpt.pt"),
            },
            f,
            indent=2,
        )
    print(f"done. best val loss = {best_val:.4f}. checkpoint -> {os.path.join(out_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
