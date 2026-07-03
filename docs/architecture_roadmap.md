# Architecture Roadmap

This page is the visual map for `gpt2-to-2026`: what the baseline model looks
like, where each modern upgrade attaches, and what should be compared in the
future interactive demo.

## Upgrade Path

```mermaid
flowchart LR
    S0["Step 0<br/>GPT-2 baseline<br/>char tokenizer, learned pos, LayerNorm, GELU MLP, MHA"]
    S1["Step 1<br/>BPE tokenizer"]
    S2["Step 2<br/>RoPE"]
    S3["Step 3<br/>RMSNorm"]
    S4["Step 4<br/>SwiGLU"]
    S5["Step 5<br/>GQA + KV cache"]
    S6["Step 6<br/>SDPA / FlashAttention"]
    S7["Step 7<br/>MoE or MLA"]
    S8["Step 8<br/>Tiny SFT"]

    S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
```

The important rule: each step changes one intended component while keeping the
rest of the experiment as fixed as possible.

## Baseline Data Flow

```mermaid
flowchart TD
    Text["Raw text"]
    Tok["Character tokenizer"]
    IDs["Token ids"]
    Batch["x, y training batches"]
    Emb["Token embedding + learned position embedding"]
    Blocks["N x GPT block"]
    Head["LM head"]
    Loss["Cross entropy loss"]
    Sample["Autoregressive generation"]

    Text --> Tok --> IDs --> Batch --> Emb --> Blocks --> Head
    Head --> Loss
    Head --> Sample
```

## Baseline GPT Block

```mermaid
flowchart TD
    X["Input x"]
    LN1["LayerNorm"]
    MHA["Causal multi-head attention"]
    Add1["Residual add"]
    LN2["LayerNorm"]
    MLP["GELU MLP<br/>Linear -> GELU -> Linear"]
    Add2["Residual add"]
    Y["Output"]

    X --> LN1 --> MHA --> Add1
    X --> Add1
    Add1 --> LN2 --> MLP --> Add2
    Add1 --> Add2
    Add2 --> Y
```

## What Each Step Changes

| Step | Component | Baseline | Upgrade | Main thing to observe |
| --- | --- | --- | --- | --- |
| 1 | Tokenizer | character-level | byte-level BPE | token count, samples, speed |
| 2 | Position | learned absolute table | RoPE on q/k | loss, long prompt behavior |
| 3 | Norm | LayerNorm | RMSNorm | params, stability, loss |
| 4 | MLP | GELU MLP | SwiGLU gated MLP | loss and sample quality |
| 5 | Attention inference | MHA, no cache | GQA + KV cache | decode tok/s, KV memory |
| 6 | Attention kernel | explicit attention | SDPA / FlashAttention | train speed, memory |
| 7 | Frontier block | dense Transformer | MoE or MLA | capability vs complexity |
| 8 | Objective | next-token LM | SFT | instruction following |

## Step Deltas

### Step 1: BPE

```mermaid
flowchart LR
    A["Raw text"] --> B0["char tokenizer"]
    A --> B1["BPE tokenizer"]
    B0 --> C0["longer token stream"]
    B1 --> C1["shorter subword token stream"]
```

### Step 2: RoPE

```mermaid
flowchart TD
    Emb["Token embedding"]
    Pos["learned position table"]
    QK["q/k vectors inside attention"]
    RoPE["RoPE rotation"]
    Attn["attention scores"]

    Emb -. "baseline adds" .-> Pos
    Pos -. "removed in Step 2" .-> Emb
    Emb --> QK --> RoPE --> Attn
```

### Step 3: RMSNorm

```mermaid
flowchart LR
    LN["LayerNorm<br/>mean + variance + gain + bias"]
    RMS["RMSNorm<br/>RMS + gain"]
    LN --> RMS
```

### Step 4: SwiGLU

```mermaid
flowchart LR
    GELU["GELU MLP<br/>Linear -> GELU -> Linear"]
    SWI["SwiGLU MLP<br/>SiLU gate * value -> Linear"]
    GELU --> SWI
```

### Step 5: GQA + KV Cache

```mermaid
flowchart TD
    Q["many query heads"]
    KV0["one key/value head per query head<br/>MHA"]
    KV1["fewer shared key/value heads<br/>GQA"]
    Cache["KV cache during generation"]
    Speed["lower decode work / memory"]

    Q --> KV0
    Q --> KV1 --> Cache --> Speed
```

## Future Interactive Demo

The eventual demo should let a reader click a step and see three synchronized
views:

- architecture diff: which block changed
- metrics diff: loss, tokens/sec, memory, checkpoint size
- behavior diff: generated samples from the same prompt

The first useful version can be a static web page backed by exported
`reports/runs/<name>/` artifacts. A later version can load checkpoints locally
and generate text interactively.
