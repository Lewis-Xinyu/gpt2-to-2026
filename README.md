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

> ⚠️ **Status: work in progress.** Building the Step-0 baseline first.
> 状态：开发中，正在搭第 0 步基线。

## The upgrade chain / 升级链

每一步 = 代码模块 + 一段「为什么」+ 消融表。Each step = a code module + a "why" note + an ablation table.

- [x] **Step 0 — Baseline**: vanilla GPT-2 (learned pos-emb, LayerNorm, GELU MLP, MHA)
- [ ] **Step 1 — BPE tokenizer** (replace char-level)
- [ ] **Step 2 — RoPE** rotary position embeddings
- [ ] **Step 3 — RMSNorm** (replace LayerNorm)
- [ ] **Step 4 — SwiGLU** FFN (replace GELU MLP)
- [ ] **Step 5 — GQA + KV cache**
- [ ] **Step 6 — better init / LR schedule, weight tying, SDPA/FlashAttention**
- [ ] **Step 7 — one frontier piece**: MoE *or* MLA (DeepSeek)
- [ ] **Step 8 — SFT**: turn it into a tiny chat model
- [ ] *(stretch)* tiny DPO/GRPO; interactive web demo

## Quickstart / 快速开始

```bash
# 1. environment (Python 3.13 venv)
python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt

# 2. prepare data (downloads ~1MB tiny-shakespeare, char-level)
.venv/bin/python data/prepare.py

# 3. train the Step-0 baseline            # (coming next)
.venv/bin/python train.py --config configs/baseline.yaml

# 4. sample from a checkpoint             # (coming next)
.venv/bin/python eval/generate.py --out_dir out/baseline
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
```

## Why this exists / 为什么做这个

Most "LLM from scratch" repos either stop at vanilla GPT-2 or hand you a finished
modern model. Few show the **delta** between them — what each modern trick buys
you, measured. This repo is that delta, made reproducible and cheap.

## Credits

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and
[karpathy/minGPT](https://github.com/karpathy/minGPT). Built as a learning project.

## License

MIT
