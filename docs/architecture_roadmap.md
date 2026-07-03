# Architecture Roadmap / 架构路线图

This page is the visual map for `gpt2-to-2026`: what the baseline model looks
like, where each modern upgrade attaches, and what should be compared in the
future interactive demo.

这一页是 `gpt2-to-2026` 的图解地图：baseline 模型长什么样、每个现代 LLM 组件改在
哪里、未来交互式 Demo 应该展示哪些对比。

## 1. Full Upgrade Path / 完整升级路径

```mermaid
flowchart LR
    S0["Step 0<br/>GPT-2 baseline<br/>char tokenizer<br/>learned pos<br/>LayerNorm<br/>GELU MLP<br/>MHA"]
    S1["Step 1<br/>BPE tokenizer<br/>subword tokens"]
    S2["Step 2<br/>RoPE<br/>rotary q/k"]
    S3["Step 3<br/>RMSNorm<br/>simpler norm"]
    S4["Step 4<br/>SwiGLU<br/>gated MLP"]
    S5["Step 5<br/>GQA + KV cache<br/>faster decode"]
    S6["Step 6<br/>SDPA / FlashAttention<br/>fused attention"]
    S7["Step 7<br/>MoE or MLA<br/>frontier block"]
    S8["Step 8<br/>Tiny SFT<br/>instruction toy"]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
```

**English:** Each step changes one intended component while keeping the rest of
the experiment as fixed as possible.

**中文：** 每一步只改变一个目标组件，其余条件尽量保持不变。这样才能判断“这次升级”到底带来了什么。

## 2. Baseline Training Flow / Baseline 训练流程

```mermaid
flowchart TD
    Text["Raw text<br/>原始文本"]
    Tok["Character tokenizer<br/>字符级 tokenizer"]
    IDs["Token ids<br/>整数 token 序列"]
    Batch["Training batches<br/>x: input, y: next token"]
    Emb["Token emb + position emb<br/>词向量 + 位置向量"]
    Blocks["GPT blocks<br/>多层 GPT Block"]
    Head["LM head<br/>预测下一个 token"]
    Loss["Cross entropy loss<br/>交叉熵损失"]
    Sample["Autoregressive generation<br/>自回归生成"]

    Text --> Tok --> IDs --> Batch --> Emb --> Blocks --> Head
    Head --> Loss
    Head --> Sample
```

**English:** GPT training is next-token prediction. Given token sequence
`x[0:T]`, the model learns to predict `x[1:T+1]`.

**中文：** GPT 训练本质上是“预测下一个 token”。给模型一段 token 序列 `x[0:T]`，
让它学习预测右移一位的序列 `x[1:T+1]`。

## 3. Baseline GPT Block / Baseline GPT Block 结构

```mermaid
flowchart TD
    X["Input x<br/>输入"]
    LN1["LayerNorm<br/>归一化"]
    MHA["Causal MHA<br/>因果多头注意力"]
    Add1["Residual add<br/>残差相加"]
    LN2["LayerNorm<br/>归一化"]
    MLP["GELU MLP<br/>Linear -> GELU -> Linear"]
    Add2["Residual add<br/>残差相加"]
    Y["Output<br/>输出"]

    X --> LN1 --> MHA --> Add1
    X --> Add1
    Add1 --> LN2 --> MLP --> Add2
    Add1 --> Add2
    Add2 --> Y
```

**English:** This is the reference block. Later upgrades replace one part at a
time: position encoding, normalization, MLP, or attention.

**中文：** 这是后续所有实验的参照组。之后的升级会逐个替换其中某一部分：位置编码、归一化、
MLP 或 attention。

## 4. Component Map / 组件对应图

```mermaid
flowchart TD
    Tokenizer["Tokenizer<br/>文本如何变成 token"]
    Embedding["Embedding<br/>token 如何变成向量"]
    Position["Position<br/>位置如何进入模型"]
    Attention["Attention<br/>token 之间如何互相看"]
    Norm["Norm<br/>激活如何保持稳定"]
    FFN["MLP / FFN<br/>每个 token 内部如何变换"]
    Output["LM head<br/>如何预测下一个 token"]

    Tokenizer --> Embedding --> Position --> Attention --> Norm --> FFN --> Output

    S1["Step 1 BPE"] --> Tokenizer
    S2["Step 2 RoPE"] --> Position
    S3["Step 3 RMSNorm"] --> Norm
    S4["Step 4 SwiGLU"] --> FFN
    S5["Step 5 GQA + cache"] --> Attention
```

**English:** This is the mental model for the whole repo. Every upgrade is tied
to a specific model subsystem.

**中文：** 这是理解整个仓库的心智模型。每一次升级都对应模型中的一个具体子系统。

## 5. What Each Step Changes / 每一步改了什么

| Step | Component / 组件 | Baseline / 原始做法 | Upgrade / 升级做法 | Observe / 观察指标 |
| --- | --- | --- | --- | --- |
| 1 | Tokenizer | character-level | byte-level BPE | token count, samples, speed |
| 2 | Position | learned absolute table | RoPE on q/k | loss, long prompt behavior |
| 3 | Norm | LayerNorm | RMSNorm | params, stability, loss |
| 4 | MLP | GELU MLP | SwiGLU gated MLP | loss and sample quality |
| 5 | Attention inference | MHA, no cache | GQA + KV cache | decode tok/s, KV memory |
| 6 | Attention kernel | explicit attention | SDPA / FlashAttention | train speed, memory |
| 7 | Frontier block | dense Transformer | MoE or MLA | capability vs complexity |
| 8 | Objective | next-token LM | SFT | instruction following |

## 6. Step Deltas / 每一步的差分图

### Step 1: BPE Tokenizer

```mermaid
flowchart LR
    A["Raw text<br/>原始文本"]
    B0["Char tokenizer<br/>一个字符一个 token"]
    B1["BPE tokenizer<br/>常见片段合成 token"]
    C0["Longer token stream<br/>序列更长"]
    C1["Shorter subword stream<br/>序列更短"]

    A --> B0 --> C0
    A --> B1 --> C1
```

**Why / 为什么：** Character tokenization is easy to debug, but real GPT-like
models use subword tokenization. BPE usually makes sequences shorter and handles
mixed text better.

字符级 tokenizer 很容易调试，但真实 GPT 类模型通常用子词 tokenizer。BPE 往往能缩短序列，
也更适合英文、符号、代码等混合文本。

### Step 2: RoPE

```mermaid
flowchart TD
    Tok["Token embedding<br/>token 向量"]
    Pos["Learned position table<br/>可学习位置表"]
    QK["q/k vectors<br/>attention 中的 q/k"]
    RoPE["RoPE rotation<br/>旋转位置编码"]
    Score["Attention score<br/>注意力分数"]

    Tok --> Pos
    Pos --> QK
    Tok --> QK
    QK --> RoPE --> Score
```

**Why / 为什么：** GPT-2 learns an absolute position vector. RoPE puts position
information into query/key geometry, which is why many modern LLMs use it.

GPT-2 使用可学习的绝对位置向量。RoPE 把位置信息放进 q/k 的几何关系里，因此被许多现代
LLM 采用。

### Step 3: RMSNorm

```mermaid
flowchart LR
    LN["LayerNorm<br/>subtract mean<br/>divide by std<br/>gain + bias"]
    RMS["RMSNorm<br/>divide by RMS<br/>gain only"]
    LN --> RMS
```

**Why / 为什么：** RMSNorm removes mean subtraction and bias. It is simpler than
LayerNorm and common in modern decoder-only LLMs.

RMSNorm 去掉了减均值和 bias，比 LayerNorm 更简单，是现代 decoder-only LLM 中常见的归一化方式。

### Step 4: SwiGLU

```mermaid
flowchart LR
    GELU["GELU MLP<br/>Linear -> GELU -> Linear"]
    Gate["Gate branch<br/>SiLU(W_gate x)"]
    Value["Value branch<br/>W_value x"]
    Mul["elementwise multiply<br/>逐元素相乘"]
    Proj["Output projection<br/>投影回 n_embd"]

    GELU --> Gate
    GELU --> Value
    Gate --> Mul
    Value --> Mul
    Mul --> Proj
```

**Why / 为什么：** SwiGLU adds a gate to the feed-forward block. The model can
select which features pass through for each token.

SwiGLU 给 FFN 加了“门控”。模型可以对每个 token 动态选择哪些特征通过。

### Step 5: GQA + KV Cache

```mermaid
flowchart TD
    Q["Query heads<br/>多个 query head"]
    KV0["MHA<br/>每个 query head 对应自己的 K/V"]
    KV1["GQA<br/>多个 query head 共享更少的 K/V"]
    Cache["KV cache<br/>缓存历史 K/V"]
    Decode["Faster decoding<br/>生成时少重复计算"]

    Q --> KV0
    Q --> KV1 --> Cache --> Decode
```

**Why / 为什么：** During generation, old tokens do not change. KV cache reuses
their key/value tensors. GQA stores fewer K/V heads, reducing cache memory.

生成时，历史 token 不会改变。KV cache 会复用历史 token 的 key/value。GQA 进一步减少 K/V
head 数量，从而降低 cache 显存。

## 7. Possible Next Improvements / 后续可以改进的点

This roadmap should become more useful as real training results come in. The
next improvements are practical and incremental:

这张路线图后面应该随着真实训练结果继续补充。比较实际的改进方向有：

### 7.1 Add Results Beside Each Step / 给每一步补实验结果

After the RTX 5070 runs are finished, each step should show:

等 RTX 5070 上的训练跑完后，每一步都应该补上：

- validation loss / 验证集 loss
- tokens per second / 训练吞吐
- generation speed with and without cache / 有无 KV cache 的生成速度
- sample outputs from the same prompt / 同一个 prompt 的生成样例

This will make the roadmap more than a concept map: readers can see whether a
component changes quality, speed, memory, or mainly code structure.

这样路线图就不只是概念图，而是能回答更具体的问题：某个组件到底影响质量、速度、显存，
还是主要改变了代码结构。

### 7.2 Add More Detailed Component Diagrams / 继续细化组件图

Some parts still deserve their own diagrams:

还有几块值得单独画得更清楚：

- **Causal attention / 因果注意力**：why future tokens are masked  
  为什么生成模型不能看到未来 token。
- **RoPE / 旋转位置编码**：how q/k pairs are rotated  
  q/k 向量是怎么按位置旋转的。
- **GQA / 分组查询注意力**：how query heads share fewer KV heads  
  多个 query head 如何共享更少的 key/value head。
- **KV cache / KV 缓存**：what is reused during token-by-token generation  
  逐 token 生成时到底复用了什么。

### 7.3 Build a Static Comparison Page / 做一个静态对比页面

The first demo does not need to load models. A useful first version can simply
read exported files from `reports/runs/<name>/` and show:

第一个 Demo 不需要加载模型。一个有用的初版可以直接读取 `reports/runs/<name>/` 中导出的文件，
展示：

- the changed module / 这一步改了哪个模块
- the config used for the run / 使用的训练配置
- the loss curve / loss 曲线
- generated samples / 生成样例
- notes about what improved or got worse / 对结果的简短分析

### 7.4 Later: Interactive Checkpoint Demo / 后续再做 checkpoint 交互 Demo

After enough checkpoints exist, a later demo can load them locally and let the
reader use the same prompt across multiple steps.

等 checkpoint 足够多之后，再做真正的本地交互 Demo：读者输入同一个 prompt，选择不同 Step 的
checkpoint，观察输出如何变化。
