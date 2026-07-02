# Step 3: RMSNorm

Step 3 replaces GPT-2's LayerNorm with RMSNorm while keeping the Step 2 RoPE
setup fixed.

## What Changes

LayerNorm normalizes by subtracting the mean and dividing by standard deviation:

```text
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm removes the mean subtraction and bias:

```text
y = x / sqrt(mean(x^2) + eps) * gamma
```

The residual layout stays pre-norm:

```text
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

## Why It Matters

RMSNorm is common in modern LLMs because it is simpler and slightly cheaper than
LayerNorm while remaining stable in large decoder-only models. In this project,
it is a clean normalization ablation:

- same BPE tokenizer as Step 1
- same RoPE position encoding as Step 2
- same model size and training schedule
- one changed config value: `norm_type: rmsnorm`

## Run

On the Windows training machine:

```bash
python train.py --config configs/rmsnorm_5070.yaml
python eval/generate.py --out_dir out/rmsnorm_5070 --start "ROMEO:" --output_file out/rmsnorm_5070/samples.txt
python scripts/plot_log.py --log out/rmsnorm_5070/log.csv --out out/rmsnorm_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rmsnorm_5070
```

Compare against `reports/runs/rope_5070/`.
