"""
baseline_gpt.py  —  Step 0 of the upgrade chain: a vanilla GPT-2.

This is the reference implementation everything else is measured against. It is
deliberately "2019 GPT-2", not a modern LLM:

    * learned absolute position embeddings   (Step 2 swaps in RoPE)
    * LayerNorm                               (Step 3 swaps in RMSNorm)
    * GELU MLP with 4x expansion              (Step 4 swaps in SwiGLU)
    * standard multi-head attention, written  (Step 5 adds GQA + KV cache,
      out EXPLICITLY (no flash attention)      Step 6 swaps in fused SDPA)

Reference: Radford et al. 2019, "Language Models are Unsupervised Multitask
Learners" (GPT-2). Architecture notes follow Vaswani et al. 2017 (the
decoder half of "Attention Is All You Need") with the GPT-2 pre-norm placement.

The code is written to be READ. Shapes are annotated as (B, T, C) where
B = batch, T = time/sequence length, C = channels/embedding dim (n_embd).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256        # max context length
    vocab_size: int = 65         # filled from data meta.pkl
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True            # GPT-2 uses bias in Linear & LayerNorm layers
    position_embedding: str = "learned"  # learned (GPT-2) or rope (Step 2)
    norm_type: str = "layernorm"          # layernorm (GPT-2) or rmsnorm (Step 3)
    mlp_type: str = "gelu"                # gelu (GPT-2) or swiglu (Step 4)


class RotaryEmbedding(nn.Module):
    """RoPE rotates q/k pairs by a position-dependent angle inside attention.

    Learned absolute embeddings add a position vector to the residual stream.
    RoPE instead changes the geometry of q and k so attention scores carry
    relative-position information. Values are left untouched.
    """

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even per-head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        pos = torch.arange(T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        cos = freqs.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=q.dtype)[None, None, :, :]
        return self._apply(q, cos, sin), self._apply(k, cos, sin)

    @staticmethod
    def _apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return x_rot.flatten(-2)


class LayerNorm(nn.Module):
    """LayerNorm with an OPTIONAL bias. (PyTorch's built-in always has bias
    unless you pass elementwise_affine=False, which also drops the gain.)

    Written by hand so the contrast with RMSNorm at Step 3 is explicit:
        LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
        RMSNorm:    y =  x / sqrt(mean(x^2) + eps)    * gamma      (no mean-subtract, no beta)
    """

    def __init__(self, n_embd: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm: normalize by RMS, no mean subtraction.

    Many modern LLMs use RMSNorm because it is simpler than LayerNorm and works
    well at scale. There is only a learned gain; no bias term is used.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x


def build_norm(config: GPTConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return LayerNorm(config.n_embd, bias=config.bias)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError(f"unknown norm_type: {config.norm_type}")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, written explicitly.

    "Explicitly" means we form the full (T, T) attention matrix and softmax it
    ourselves instead of calling a fused kernel. It is slower and uses O(T^2)
    memory, but every step is visible. Step 6 replaces the marked block with
    F.scaled_dot_product_attention and measures the speedup.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.rotary = RotaryEmbedding(self.head_dim) if config.position_embedding == "rope" else None

        # One projection produces q, k, v for ALL heads at once, then we split.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection that mixes the heads back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular causal mask (1 = keep, 0 = mask). Not a parameter, so
        # we register it as a buffer (moves with .to(device), not trained).
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to q, k, v and reshape to (B, n_head, T, head_dim).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.rotary is not None:
            q, k = self.rotary(q, k)

        # ---- explicit scaled dot-product attention -------------------------
        # scores[b,h,i,j] = how much query i attends to key j   -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # causal mask: a token may not look at future tokens (j > i).
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                              # (B, nh, T, head_dim)
        # --------------------------------------------------------------------

        # Re-assemble all heads side by side: (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Linear, 4x wide."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class SwiGLUMLP(nn.Module):
    """Gated feed-forward network used by many modern decoder-only LLMs.

    GPT-2 uses Linear -> GELU -> Linear. SwiGLU uses a learned gate:
        SiLU(W_gate x) * W_value x

    The hidden size is about 2/3 of the GELU MLP's 4x width, which keeps the
    parameter count in the same neighborhood while adding the gating mechanism.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = math.ceil((8 * config.n_embd / 3) / 8) * 8
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_value = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w_gate(x)) * self.w_value(x)
        return self.dropout(self.c_proj(x))


def build_mlp(config: GPTConfig) -> nn.Module:
    if config.mlp_type == "gelu":
        return MLP(config)
    if config.mlp_type == "swiglu":
        return SwiGLUMLP(config)
    raise ValueError(f"unknown mlp_type: {config.mlp_type}")


class Block(nn.Module):
    """A transformer block with PRE-norm residual connections (GPT-2 style):

        x = x + attn(norm(x))
        x = x + mlp(norm(x))

    Pre-norm (norm inside the residual branch) trains far more stably than the
    original post-norm of Vaswani et al., which is why every modern LLM uses it.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config)
        self.mlp = build_mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        assert config.position_embedding in {"learned", "rope"}
        assert config.norm_type in {"layernorm", "rmsnorm"}
        assert config.mlp_type in {"gelu", "swiglu"}
        self.config = config
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token emb
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f=build_norm(config),
        )
        if config.position_embedding == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and the output projection share the
        # same matrix (Press & Wolf 2016). Used by GPT-2; saves vocab*n_embd params.
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 initialization.
        self.apply(self._init_weights)
        # Scale down the residual-path projections by 1/sqrt(2*n_layer) so the
        # residual stream variance doesn't blow up with depth (GPT-2 trick).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n -= self.transformer.wpe.weight.numel()  # wte is tied, counted once
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"

        tok_emb = self.transformer.wte(idx)   # (B, T, C) token meanings
        if self.config.position_embedding == "learned":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)   # (T, C)    "where am I"
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)          # (B, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        # Inference shortcut: only compute logits for the LAST position.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with decoupled weight decay applied only to 2D matrices.
        Biases and LayerNorm gains (1D tensors) are excluded from weight decay."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively sample max_new_tokens, feeding predictions back in."""
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (the model's memory).
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
