# Step 4: SwiGLU

Step 4 replaces GPT-2's GELU feed-forward network with a gated SwiGLU MLP while
keeping the Step 3 setup fixed.

## What Changes

GPT-2's MLP is:

```text
Linear -> GELU -> Linear
```

SwiGLU is:

```text
SiLU(W_gate x) * W_value x -> Linear
```

The multiplication is the gate. It lets the network decide which features should
pass through the feed-forward block for each token.

## Why It Matters

In decoder-only Transformers, the MLP block is a major share of the parameters
and compute. Modern LLMs often use gated MLP variants because they improve
capacity and training behavior at similar scale.

In this project, SwiGLU is a clean FFN ablation:

- same BPE tokenizer
- same RoPE position encoding
- same RMSNorm normalization
- same training schedule
- one changed config value: `mlp_type: swiglu`

## Run

On the Windows training machine:

```bash
python train.py --config configs/swiglu_5070.yaml
python eval/generate.py --out_dir out/swiglu_5070 --start "ROMEO:" --output_file out/swiglu_5070/samples.txt
python scripts/plot_log.py --log out/swiglu_5070/log.csv --out out/swiglu_5070/loss_curve.png
python scripts/export_run.py --out_dir out/swiglu_5070
```

Compare against `reports/runs/rmsnorm_5070/`.
