# Step 0: Vanilla GPT-2 Baseline

Step 0 is the reference model for the whole upgrade chain. The goal is not to
match GPT-2 small's 124M parameter scale. The goal is to keep the GPT-2 design
recognizable while making training cheap enough that every later upgrade can be
measured under the same budget.

## What It Implements

- decoder-only Transformer language modeling
- learned token embeddings
- learned absolute position embeddings
- pre-norm residual blocks
- multi-head causal self-attention
- GELU MLP with 4x hidden expansion
- LayerNorm with GPT-2-style optional bias
- tied input embedding and output LM head
- AdamW with cosine decay and warmup

## What It Intentionally Does Not Use Yet

- BPE tokenizer
- RoPE
- RMSNorm
- SwiGLU
- grouped-query attention
- KV cache
- fused SDPA or FlashAttention
- SFT or preference tuning

Those omissions are the point: each one becomes a later controlled upgrade.

## Why Six Layers

The baseline uses a small proxy shape by default:

```text
n_layer = 6
n_head  = 6
n_embd  = 384
block   = 256
```

This is GPT-2-style, not GPT-2-size. A true GPT-2 small shape would be closer to
12 layers, 12 heads, 768 hidden size, 1024 context, and roughly 124M parameters.
That is useful as a later scale-up check, but it is too expensive as the default
for every ablation.

## Evidence To Collect

For the first full CUDA run, export:

- final and best validation loss
- training loss curve
- tokens per second
- generated samples from a fixed prompt
- GPU name, dtype, and training config

These artifacts live under `reports/runs/baseline_5070/` after running
`scripts/export_run.py`.
