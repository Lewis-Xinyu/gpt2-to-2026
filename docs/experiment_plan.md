# Experiment Plan

This repository is organized as a sequence of controlled upgrades from a small
GPT-2-style baseline toward modern LLM components. Every step should keep the
same dataset, seed, training budget, and reporting format unless the experiment
explicitly studies one of those variables.

## Workflow

- Develop code and documentation on the Mac.
- Train on the RTX 5070 Windows laptop.
- Keep checkpoints and raw tokenized data under ignored paths such as `out/` and
  `data/**/*.bin`.
- Export lightweight evidence into `reports/runs/<run_name>/` for Git:
  configs, logs, summaries, and generated samples.

## Step 0 Baseline

Run on the Windows training machine:

```bash
python scripts/check_env.py
python data/prepare.py
python train.py --config configs/baseline_5070.yaml
python eval/generate.py --out_dir out/baseline_5070 --start "ROMEO:" --output_file out/baseline_5070/samples.txt
python scripts/plot_log.py --log out/baseline_5070/log.csv --out out/baseline_5070/loss_curve.png
python scripts/export_run.py --out_dir out/baseline_5070
```

Commit the exported `reports/runs/baseline_5070/` files, not the checkpoint.

## Metrics To Record

- train loss and validation loss
- best validation loss and iteration
- tokens per second
- total training time
- GPU name and VRAM
- generation samples from a fixed prompt
- loss curve image

## Step 1 BPE

Run after Step 0 has a clean baseline result:

```bash
python data/prepare_bpe.py --vocab_size 8000
python train.py --config configs/bpe_5070.yaml
python eval/generate.py --out_dir out/bpe_5070 --start "ROMEO:" --output_file out/bpe_5070/samples.txt
python scripts/plot_log.py --log out/bpe_5070/log.csv --out out/bpe_5070/loss_curve.png
python scripts/export_run.py --out_dir out/bpe_5070
```

Keep the model shape fixed so the tokenizer is the only intended variable.

## Step 2 RoPE

Run after Step 1 has a clean BPE result:

```bash
python train.py --config configs/rope_5070.yaml
python eval/generate.py --out_dir out/rope_5070 --start "ROMEO:" --output_file out/rope_5070/samples.txt
python scripts/plot_log.py --log out/rope_5070/log.csv --out out/rope_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rope_5070
```

Compare against `bpe_5070`; keep tokenizer, model size, and training budget fixed.

## Step 3 RMSNorm

Run after Step 2 has a clean RoPE result:

```bash
python train.py --config configs/rmsnorm_5070.yaml
python eval/generate.py --out_dir out/rmsnorm_5070 --start "ROMEO:" --output_file out/rmsnorm_5070/samples.txt
python scripts/plot_log.py --log out/rmsnorm_5070/log.csv --out out/rmsnorm_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rmsnorm_5070
```

Compare against `rope_5070`; keep tokenizer, RoPE, model size, and training budget fixed.

## Step 4 SwiGLU

Run after Step 3 has a clean RMSNorm result:

```bash
python train.py --config configs/swiglu_5070.yaml
python eval/generate.py --out_dir out/swiglu_5070 --start "ROMEO:" --output_file out/swiglu_5070/samples.txt
python scripts/plot_log.py --log out/swiglu_5070/log.csv --out out/swiglu_5070/loss_curve.png
python scripts/export_run.py --out_dir out/swiglu_5070
```

Compare against `rmsnorm_5070`; keep tokenizer, RoPE, RMSNorm, model size, and training budget fixed.

## Step 5 GQA + KV Cache

Run after Step 4 has a clean SwiGLU result:

```bash
python train.py --config configs/gqa_5070.yaml
python eval/generate.py --out_dir out/gqa_5070 --start "ROMEO:" --output_file out/gqa_5070/samples.txt --use_cache
python scripts/plot_log.py --log out/gqa_5070/log.csv --out out/gqa_5070/loss_curve.png
python scripts/export_run.py --out_dir out/gqa_5070
```

Compare against `swiglu_5070`; for decode speed, generate from both checkpoints
with and without `--use_cache`.
