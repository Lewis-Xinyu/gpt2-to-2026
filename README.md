# gpt2-to-2026

> **From GPT-2 to modern LLMs: a hands-on roadmap of architecture upgrades,
> readable PyTorch code, and controlled ablation experiments.**
>
> 从 GPT-2 到现代 LLM：用最小 PyTorch 实现、图解文档和消融实验，理解大模型架构是如何一步步演进的。

`gpt2-to-2026` is not a production LLM repo and not a personal note dump. It is a
small, reproducible learning lab: start from a GPT-2-style baseline, then add one
modern LLM component at a time under the same tiny model, same data, same seed,
and same reporting format.

The goal is to make the delta visible: what changes in the code, why modern
models use it, and what it does to loss, throughput, memory, and generation.

## Roadmap

```text
GPT-2 baseline
  -> BPE tokenizer
  -> RoPE
  -> RMSNorm
  -> SwiGLU
  -> GQA + KV cache
  -> SDPA / FlashAttention
  -> one frontier block: MoE or MLA
  -> tiny SFT
  -> stretch: DPO/GRPO, RAG/Agent demo
```

Each step should include:

- readable implementation
- config for a controlled experiment
- short explanation of why it exists
- tests for the new path
- exported run artifacts once trained

## Progress

| Step | Upgrade | Status | Notes |
| --- | --- | --- | --- |
| 0 | Vanilla GPT-2 baseline | Code ready | Learned pos-emb, LayerNorm, GELU MLP, MHA |
| 1 | BPE tokenizer | Code ready | Replaces char-level tokenization |
| 2 | RoPE | Code ready | Replaces learned absolute position table |
| 3 | RMSNorm | Code ready | Replaces LayerNorm |
| 4 | SwiGLU | Code ready | Replaces GELU MLP |
| 5 | GQA + KV cache | Code ready | Adds fewer KV heads and cached generation |
| 6 | SDPA / FlashAttention | Planned | Fused attention and speed comparison |
| 7 | MoE or MLA | Planned | One frontier architecture block |
| 8 | Tiny SFT | Planned | Turn the tiny LM into an instruction-following toy |

First full CUDA training runs are still pending. The current focus is preparing a
clean experiment suite before running everything on a consumer NVIDIA GPU.

## Who Is This For?

- CS students who want to understand LLMs from code, not only papers
- deep learning learners who know basic PyTorch and want a next project
- developers curious about what changed between GPT-2 and modern open LLMs
- anyone who prefers small, runnable ablations over giant training systems

## What Makes It Different?

Most open LLM repos are either compact GPT baselines or production-scale training
systems. This repo focuses on the missing middle: a small upgrade path where each
modern component is isolated, implemented, tested, and eventually benchmarked.

The project is intentionally small enough to run on a single consumer GPU, but
structured like a real experiment lab:

- `configs/` keeps one YAML per ablation
- `docs/` explains the purpose of every step
- `tests/` protects model invariants like causality and cache correctness
- `reports/runs/` stores lightweight results, not giant checkpoints

## Start Here

Read the steps:

- [Architecture Roadmap](docs/architecture_roadmap.md)
- [Step 0: Vanilla GPT-2 Baseline](docs/step0_baseline.md)
- [Step 1: BPE Tokenizer](docs/step1_bpe.md)
- [Step 2: RoPE](docs/step2_rope.md)
- [Step 3: RMSNorm](docs/step3_rmsnorm.md)
- [Step 4: SwiGLU](docs/step4_swiglu.md)
- [Step 5: GQA + KV Cache](docs/step5_gqa_kv_cache.md)

Run local tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

Prepare the tiny Shakespeare char dataset and train Step 0:

```bash
.venv/bin/python data/prepare.py
.venv/bin/python train.py --config configs/baseline.yaml
.venv/bin/python eval/generate.py --out_dir out/baseline --start "ROMEO:"
```

## CUDA Experiment Flow

For the RTX 5070 / NVIDIA training machine:

```bash
python scripts/check_env.py
python data/prepare.py
python train.py --config configs/baseline_5070.yaml
python eval/generate.py --out_dir out/baseline_5070 --start "ROMEO:" --output_file out/baseline_5070/samples.txt
python scripts/plot_log.py --log out/baseline_5070/log.csv --out out/baseline_5070/loss_curve.png
python scripts/export_run.py --out_dir out/baseline_5070
```

Then continue with:

```bash
python data/prepare_bpe.py --vocab_size 8000
python train.py --config configs/bpe_5070.yaml
python train.py --config configs/rope_5070.yaml
python train.py --config configs/rmsnorm_5070.yaml
python train.py --config configs/swiglu_5070.yaml
python train.py --config configs/gqa_5070.yaml
```

Full run commands live in [docs/experiment_plan.md](docs/experiment_plan.md).

## Repo Layout

```text
model/      readable model components
data/       dataset preparation scripts
configs/    one YAML per ablation
eval/       sampling and generation utilities
docs/       explanations for each step
scripts/    environment checks, plotting, export helpers
reports/    lightweight run artifacts that can be committed
tests/      model and config behavior tests
```

## Related Work

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and
[karpathy/minGPT](https://github.com/karpathy/minGPT), but the output is an
upgrade chain rather than a single baseline.

This is also not a replacement for large training systems such as EleutherAI
GPT-Neo. Those projects focus on distributed training and released model
weights; this repo focuses on small, controlled ablations that explain how each
modern LLM component changes the GPT-2 baseline.

## License

MIT
