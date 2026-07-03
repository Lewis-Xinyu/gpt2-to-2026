# gpt2-to-2026

> **From GPT-2 to modern LLMs: a hands-on roadmap of architecture upgrades,
> readable PyTorch code, and controlled ablation experiments.**
>
> **从 GPT-2 到现代 LLM：用最小 PyTorch 实现、图解文档和消融实验，理解大模型架构是如何一步步演进的。**

`gpt2-to-2026` is a small, reproducible learning lab. It starts from a
GPT-2-style baseline and adds one modern LLM component at a time under the same
tiny model, same data, same seed, and same reporting format.

`gpt2-to-2026` 是一个小型、可复现的大模型学习实验室。它从 GPT-2 风格的 baseline
出发，每次只加入一个现代 LLM 组件，并尽量保持模型大小、数据、随机种子和实验记录格式一致。

The goal is not to train a strong chatbot. The goal is to make the **delta**
visible: what changes in the code, why modern models use it, and what it does to
loss, throughput, memory, and generation.

本项目的目标不是训练一个强聊天机器人，而是把每一次架构演进的 **变化量** 讲清楚：
代码改了哪里、为什么现代模型会这样做、它对 loss、速度、显存和生成效果有什么影响。

## Visual Roadmap / 图解路线图

Start with the visual guide:

先从图解导览开始：

- [Architecture Roadmap / 架构路线图](docs/architecture_roadmap.md)

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

Each step should include readable code, a controlled config, a short explanation,
tests, and exported run artifacts once trained.

每一步都应该包含：可读代码、可控实验配置、简短解释、测试，以及训练完成后的轻量实验结果。

## Progress / 当前进度

| Step | Upgrade | Status | What changes / 改了什么 |
| --- | --- | --- | --- |
| 0 | Vanilla GPT-2 baseline | Code ready | char tokenizer, learned pos-emb, LayerNorm, GELU MLP, MHA |
| 1 | BPE tokenizer | Code ready | character tokens -> byte-level subword tokens |
| 2 | RoPE | Code ready | learned position table -> rotary q/k position encoding |
| 3 | RMSNorm | Code ready | LayerNorm -> RMS-only normalization |
| 4 | SwiGLU | Code ready | GELU MLP -> gated MLP |
| 5 | GQA + KV cache | Code ready | MHA/no cache -> fewer KV heads + cached generation |
| 6 | SDPA / FlashAttention | Planned | explicit attention -> fused attention kernels |
| 7 | MoE or MLA | Planned | dense block -> one frontier architecture block |
| 8 | Tiny SFT | Planned | next-token LM -> toy instruction-following model |

First full CUDA training runs are still pending. The current focus is preparing a
clean experiment suite before running everything on a consumer NVIDIA GPU.

完整 CUDA 训练结果还没跑。当前阶段是在消费级 NVIDIA GPU 上训练前，把实验代码、配置、
文档和测试准备干净。

## Who Is This For? / 适合谁？

- CS students who want to understand LLMs from code, not only papers  
  想从代码理解 LLM，而不是只看论文的计算机学生
- deep learning learners who know basic PyTorch and want a next project  
  学过基础 PyTorch，想做一个更系统项目的深度学习学习者
- developers curious about what changed between GPT-2 and modern open LLMs  
  想知道 GPT-2 和现代开源 LLM 到底差在哪些组件上的开发者
- anyone who prefers small, runnable ablations over giant training systems  
  更喜欢小型可运行消融实验，而不是上来就看巨型训练系统的人

## What Makes It Different? / 这个项目有什么不同？

Most open LLM repos are either compact GPT baselines or production-scale training
systems. This repo focuses on the missing middle: a small upgrade path where each
modern component is isolated, implemented, tested, and eventually benchmarked.

很多开源 LLM 仓库要么是极简 GPT baseline，要么是生产级大规模训练系统。本项目关注中间地带：
用一个小模型，把每个现代组件单独拆出来，实现、测试、解释，并最终做实验对比。

The project is intentionally small enough to run on a single consumer GPU, but
structured like a real experiment lab:

项目故意保持在单张消费级 GPU 可运行的规模，但组织方式尽量像真正的实验工程：

- `configs/`: one YAML per ablation / 每个消融实验一个 YAML
- `docs/`: visual explanations for each step / 每一步的图解说明
- `tests/`: model invariants like causality and cache correctness / 因果 mask、KV cache 等关键行为测试
- `reports/runs/`: lightweight results, not giant checkpoints / 只提交轻量结果，不提交巨大权重

## Start Here / 从这里开始

Read the visual overview and step notes:

先读图解总览和每一步说明：

- [Architecture Roadmap / 架构路线图](docs/architecture_roadmap.md)
- [Step 0: Vanilla GPT-2 Baseline](docs/step0_baseline.md)
- [Step 1: BPE Tokenizer](docs/step1_bpe.md)
- [Step 2: RoPE](docs/step2_rope.md)
- [Step 3: RMSNorm](docs/step3_rmsnorm.md)
- [Step 4: SwiGLU](docs/step4_swiglu.md)
- [Step 5: GQA + KV Cache](docs/step5_gqa_kv_cache.md)

Run local tests:

运行本地测试：

```bash
.venv/bin/python -m unittest discover -s tests
```

Prepare the tiny Shakespeare char dataset and train Step 0:

准备 tiny Shakespeare 字符级数据，并训练 Step 0：

```bash
.venv/bin/python data/prepare.py
.venv/bin/python train.py --config configs/baseline.yaml
.venv/bin/python eval/generate.py --out_dir out/baseline --start "ROMEO:"
```

## CUDA Experiment Flow / CUDA 实验流程

For the RTX 5070 / NVIDIA training machine:

在 RTX 5070 / NVIDIA 训练机上：

```bash
python scripts/check_env.py
python data/prepare.py
python train.py --config configs/baseline_5070.yaml
python eval/generate.py --out_dir out/baseline_5070 --start "ROMEO:" --output_file out/baseline_5070/samples.txt
python scripts/plot_log.py --log out/baseline_5070/log.csv --out out/baseline_5070/loss_curve.png
python scripts/export_run.py --out_dir out/baseline_5070
```

Then continue with the controlled upgrade runs:

然后继续跑后续可控升级实验：

```bash
python data/prepare_bpe.py --vocab_size 8000
python train.py --config configs/bpe_5070.yaml
python train.py --config configs/rope_5070.yaml
python train.py --config configs/rmsnorm_5070.yaml
python train.py --config configs/swiglu_5070.yaml
python train.py --config configs/gqa_5070.yaml
```

Full run commands live in [docs/experiment_plan.md](docs/experiment_plan.md).

完整实验命令见 [docs/experiment_plan.md](docs/experiment_plan.md)。

## Future Demo / 未来交互 Demo

After the training runs are exported, this project should grow a small visual
demo where readers can click a step and compare:

等训练结果导出后，本项目适合做一个小型可视化 Demo，让读者点击每一步并对比：

- architecture diff / 架构差异
- validation loss and throughput / 验证 loss 与吞吐
- KV cache memory and decode speed / KV cache 显存与解码速度
- generated samples from the same prompt / 同一个 prompt 下的生成样例

## Repo Layout / 目录结构

```text
model/      readable model components / 可读模型组件
data/       dataset preparation scripts / 数据准备脚本
configs/    one YAML per ablation / 每个消融一个配置
eval/       sampling and generation utilities / 采样与生成工具
docs/       visual explanations for each step / 每一步图解说明
scripts/    environment checks, plotting, export helpers / 环境检查、画图、导出脚本
reports/    lightweight run artifacts / 可提交的轻量实验结果
tests/      model and config behavior tests / 模型与配置行为测试
```

## Related Work / 相关项目

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and
[karpathy/minGPT](https://github.com/karpathy/minGPT), but the output is an
upgrade chain rather than a single baseline.

本项目受 [nanoGPT](https://github.com/karpathy/nanoGPT) 和
[minGPT](https://github.com/karpathy/minGPT) 启发，但目标不是再做一个单一 baseline，
而是做一条可运行的架构升级链。

This is also not a replacement for large training systems such as EleutherAI
GPT-Neo. Those projects focus on distributed training and released model
weights; this repo focuses on small, controlled ablations that explain how each
modern LLM component changes the GPT-2 baseline.

它也不是 EleutherAI GPT-Neo 这类大型训练系统的替代品。那些项目关注分布式训练和发布权重；
本项目关注小型、可控、可解释的消融实验。

## License / 许可证

MIT
