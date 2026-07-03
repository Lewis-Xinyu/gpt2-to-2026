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

If you are new to GPT-style models, read these pages in order. They are written
as a learning path, not as isolated notes.

如果你刚开始系统学习 GPT 类模型，建议按下面顺序读。每一节都对应一个明确的问题：
“这一步为什么出现？它改了模型哪里？代码里怎么实现？”

1. [Architecture Roadmap / 架构路线图](docs/architecture_roadmap.md)  
   A visual overview of the whole project: data flow, GPT block structure, and
   where each upgrade attaches.  
   先看总图：文本如何变成 token，GPT block 里面有哪些模块，每一步升级分别改在哪里。

2. [Step 0: Vanilla GPT-2 Baseline](docs/step0_baseline.md)  
   Build the reference model: learned position embeddings, LayerNorm, GELU MLP,
   causal multi-head attention, and next-token training.  
   搭出参照组：可学习位置编码、LayerNorm、GELU MLP、因果多头注意力，以及 next-token 训练。

3. [Step 1: BPE Tokenizer](docs/step1_bpe.md)  
   Replace character tokens with byte-level BPE so the tiny model is closer to
   how real GPT-style models consume text.  
   把字符级 token 换成 byte-level BPE，让模型处理文本的方式更接近真实 GPT。

4. [Step 2: RoPE](docs/step2_rope.md)  
   Move position information from a learned absolute table into query/key
   rotations inside attention.  
   把位置信息从“可学习位置表”挪到 attention 的 q/k 旋转里，这是现代 LLM 常见做法。

5. [Step 3: RMSNorm](docs/step3_rmsnorm.md)  
   Replace LayerNorm with a simpler normalization used by many decoder-only LLMs.  
   用 RMSNorm 替代 LayerNorm，理解现代 decoder-only LLM 为什么常用更轻量的归一化。

6. [Step 4: SwiGLU](docs/step4_swiglu.md)  
   Replace GPT-2's GELU feed-forward network with a gated MLP.  
   把 GPT-2 的 GELU MLP 换成带门控的 SwiGLU，观察 FFN 模块如何演进。

7. [Step 5: GQA + KV Cache](docs/step5_gqa_kv_cache.md)  
   Add grouped-query attention and cached generation, mainly to study inference
   speed and KV memory.  
   加入 GQA 和 KV cache，重点观察生成速度和 KV cache 显存，而不是只看 loss。

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

## Future Improvements / 后续可以改进的点

The current repo already has the code path for Steps 0-5. After the CUDA runs
are finished, the next useful improvements are:

当前仓库已经有 Step 0-5 的代码路径。等 CUDA 训练结果跑完后，比较值得继续做的是：

- Add result tables to each step page: validation loss, tokens/sec, training
  time, and sample outputs.  
  在每个 Step 文档里补结果表：验证 loss、tokens/sec、训练耗时和生成样例。

- Add clearer diagrams for each component, especially attention, RoPE, GQA, and
  KV cache.  
  继续补图，尤其是 attention、RoPE、GQA 和 KV cache，这些最容易看懵。

- Build a small static web page from `reports/runs/<name>/`, so readers can
  click a step and compare the model before and after the upgrade.  
  基于 `reports/runs/<name>/` 做一个静态网页，让读者点击某一步就能看到升级前后的结构、
  指标和生成样例。

- Later, add an interactive local demo that loads checkpoints and generates text
  with the same prompt across different steps.  
  更后面可以做本地交互 Demo：加载不同 checkpoint，用同一个 prompt 对比生成效果。

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
