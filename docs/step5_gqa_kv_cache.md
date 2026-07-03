# Step 5: GQA + KV Cache

Step 5 changes attention for inference efficiency. It introduces grouped-query
attention and a cached generation path.

## What Changes

Standard multi-head attention gives every query head its own key/value head:

```text
n_q_heads = n_kv_heads
```

Grouped-query attention keeps all query heads but shares fewer key/value heads:

```text
n_q_heads = 6
n_kv_heads = 2
```

During attention, each key/value head is repeated across a group of query heads.
During generation, the KV cache stores fewer key/value heads, which reduces cache
memory and can improve decode speed.

## KV Cache

Without a cache, generation recomputes the whole context for every new token.
With a cache, each layer stores previous K/V tensors and only computes the new
token's K/V on the next step.

This project keeps training simple with the normal full-sequence forward pass.
The cache is used by generation:

```bash
python eval/generate.py --out_dir out/gqa_5070 --start "ROMEO:" --use_cache
```

## Run

On the Windows training machine:

```bash
python train.py --config configs/gqa_5070.yaml
python eval/generate.py --out_dir out/gqa_5070 --start "ROMEO:" --output_file out/gqa_5070/samples.txt --use_cache
python scripts/plot_log.py --log out/gqa_5070/log.csv --out out/gqa_5070/loss_curve.png
python scripts/export_run.py --out_dir out/gqa_5070
```

Compare against `reports/runs/swiglu_5070/`. For generation speed, run both
`swiglu_5070` and `gqa_5070` with and without `--use_cache`.
