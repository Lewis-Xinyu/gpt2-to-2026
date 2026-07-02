# Step 2: RoPE

Step 2 replaces GPT-2's learned absolute position embeddings with rotary
position embeddings inside self-attention.

## What Changes

Baseline GPT-2 adds a learned vector to each token embedding:

```text
x = token_embedding + learned_position_embedding
```

RoPE removes that learned position table and rotates each attention head's query
and key vectors by a deterministic position-dependent angle:

```text
q, k = rope(q, k)
attention = softmax(q @ k.T / sqrt(head_dim))
```

Values are not rotated. The residual stream only receives token embeddings.

## Why It Matters

RoPE is a standard modern LLM component because it injects relative-position
structure directly into attention scores. It also avoids tying the model as
tightly to a learned absolute position table.

In this project, RoPE is a clean first architecture upgrade after BPE:

- same tokenizer as Step 1
- same model width/depth
- same training schedule
- one changed config value: `position_embedding: rope`

## Run

On the Windows training machine, after preparing the BPE data:

```bash
python train.py --config configs/rope_5070.yaml
python eval/generate.py --out_dir out/rope_5070 --start "ROMEO:" --output_file out/rope_5070/samples.txt
python scripts/plot_log.py --log out/rope_5070/log.csv --out out/rope_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rope_5070
```

Compare against `reports/runs/bpe_5070/`, not the char-level baseline.
