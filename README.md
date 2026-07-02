# gpt2-to-2026

> **Modernize GPT-2 into a 2026-era LLM, one readable, benchmarked commit at a time.**
> 从 GPT-2 到 2026：把每一个现代 LLM 的改进，做成一次可读、可复现、带消融实验的升级。

`gpt2-to-2026` is a from-scratch teaching project. It starts from the simplest
working GPT-2-style baseline and then applies the architectural upgrades that
separate a 2018 GPT-2 from a 2026 open LLM — **each as a self-contained step with
a before/after ablation on the same tiny model, same data, same seed**. Everything
runs on a single consumer GPU (and on Apple-Silicon MPS).

这是一个**从零手写**的教学项目。它从最朴素、能跑通的 GPT-2 基线出发，逐个加入让
现代 LLM 区别于 GPT-2 的改进——**每一步都是自包含的，并在同一个小模型 / 同一份数据
/ 同一个随机种子上做前后消融对比**。全程可在一张消费级 GPU（含 Apple Silicon MPS）上复现。

> ⚠️ **Status: work in progress.** Step-0 baseline code is in place; first full
> CUDA training run is next.
> 状态：开发中，Step 0 基线代码已搭好，下一步是在 CUDA 机器上完成首次完整训练。

## The upgrade chain / 升级链

每一步 = 代码模块 + 一段「为什么」+ 消融表。Each step = a code module + a "why" note + an ablation table.

- [x] **Step 0 — Baseline**: vanilla GPT-2 (learned pos-emb, LayerNorm, GELU MLP, MHA)
- [ ] **Step 1 — BPE tokenizer** (replace char-level)
- [ ] **Step 2 — RoPE** rotary position embeddings
- [ ] **Step 3 — RMSNorm** (replace LayerNorm)
- [ ] **Step 4 — SwiGLU** FFN (replace GELU MLP)
- [ ] **Step 5 — GQA + KV cache**
- [ ] **Step 6 — SDPA/FlashAttention + compile/performance cleanup**
- [ ] **Step 7 — one frontier piece**: MoE *or* MLA (DeepSeek)
- [ ] **Step 8 — SFT**: turn it into a tiny chat model
- [ ] *(stretch)* tiny DPO/GRPO; interactive web demo

## Quickstart / 快速开始

```bash
# 1. environment (Python 3.11/3.12 recommended)
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

# 2. prepare data (downloads ~1MB tiny-shakespeare, char-level)
.venv/bin/python data/prepare.py

# 3. train the Step-0 baseline
.venv/bin/python train.py --config configs/baseline.yaml

# 4. sample from a checkpoint
.venv/bin/python eval/generate.py --out_dir out/baseline
```

For the RTX 5070 Windows training machine:

```bash
python scripts/check_env.py
python data/prepare.py
python train.py --config configs/baseline_5070.yaml
python eval/generate.py --out_dir out/baseline_5070 --start "ROMEO:" --output_file out/baseline_5070/samples.txt
python scripts/plot_log.py --log out/baseline_5070/log.csv --out out/baseline_5070/loss_curve.png
python scripts/export_run.py --out_dir out/baseline_5070
```

Run the local unit tests before pushing code to the training machine:

```bash
python -m unittest discover -s tests
```

Step 1 BPE tokenizer run:

```bash
python data/prepare_bpe.py --vocab_size 8000
python train.py --config configs/bpe_5070.yaml
python eval/generate.py --out_dir out/bpe_5070 --start "ROMEO:" --output_file out/bpe_5070/samples.txt
python scripts/plot_log.py --log out/bpe_5070/log.csv --out out/bpe_5070/loss_curve.png
python scripts/export_run.py --out_dir out/bpe_5070
```

Step 2 RoPE run:

```bash
python train.py --config configs/rope_5070.yaml
python eval/generate.py --out_dir out/rope_5070 --start "ROMEO:" --output_file out/rope_5070/samples.txt
python scripts/plot_log.py --log out/rope_5070/log.csv --out out/rope_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rope_5070
```

Step 3 RMSNorm run:

```bash
python train.py --config configs/rmsnorm_5070.yaml
python eval/generate.py --out_dir out/rmsnorm_5070 --start "ROMEO:" --output_file out/rmsnorm_5070/samples.txt
python scripts/plot_log.py --log out/rmsnorm_5070/log.csv --out out/rmsnorm_5070/loss_curve.png
python scripts/export_run.py --out_dir out/rmsnorm_5070
```

## Repo layout / 目录结构

```
model/      # readable, standalone components (baseline_gpt.py, rope.py, ...)
tokenizer/  # BPE training & encoding (introduced at Step 1)
data/       # dataset preparation scripts
configs/    # one YAML per ablation (reproducibility)
eval/       # sampling, loss/throughput benchmarks
docs/       # the "why" write-up + ablation tables for each step
demo/       # Colab notebook + (stretch) web visualization
reports/    # lightweight run artifacts that can be committed
```

## Why this exists / 为什么做这个

Most open LLM repos are either production-scale training systems or compact
from-scratch baselines. This project focuses on the missing middle: a tiny,
readable, step-by-step modernization path from GPT-2 to today's LLM architecture,
where every upgrade is measured in isolation.

## Related work / 相关项目

This project is inspired by small readable baselines such as nanoGPT/minGPT, but
its output is an upgrade chain rather than a single baseline. It is also not a
replacement for large training systems such as EleutherAI GPT-Neo: those projects
focus on distributed training and released model weights, while this repo focuses
on small, controlled ablations that explain how each modern LLM component changes
the baseline.

## Credits

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and
[karpathy/minGPT](https://github.com/karpathy/minGPT). Built as a learning project.

## License

MIT
