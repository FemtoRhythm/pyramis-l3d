"""Traditional MLA (Multi-head Latent Attention) baseline.

DeepSeek-V2 风格: 把 KV 压缩到低秩 latent, 再经上投影重建 K/V。
KV cache 只存 latent (每 token latent_dim 维), 相比全量 KV 省显存。
全稠密注意力(所有 token 互见), 用于对照 Pyramis-L3D 的稀疏路由。

自包含实现(仅依赖 torch), 与 Pyramis-L3D 共享相同的 RoPE / RMSNorm / SwiGLU MLP,
保证对比只聚焦在注意力机制上。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE 工具 (与 Pyramis-L3D 一致)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(positions.reshape(-1).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos.view(*positions.shape, head_dim), sin.view(*positions.shape, head_dim)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.unsqueeze(1) + _rotate_half(x) * sin.unsqueeze(1)


# ---------------------------------------------------------------------------
# MLA 注意力
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, latent_dim, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.rope_theta = rope_theta
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        # latent KV 压缩: down -> latent, up -> K / V
        self.down_kv = nn.Linear(d_model, latent_dim, bias=False)
        self.up_k = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.up_v = nn.Linear(latent_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        c_kv = self.down_kv(hidden)  # [B, T, latent_dim]
        k = self.up_k(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.up_v(c_kv).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cos_sin(position_ids, self.head_dim, self.rope_theta, hidden.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.tril(torch.ones(T, T, device=hidden.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLABlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, latent_dim, intermediate_size, eps=1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.self_attn = MLAAttention(d_model, n_heads, head_dim, latent_dim)
        self.post_attention_layernorm = nn.RMSNorm(d_model, eps=eps)
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = residual + self.self_attn(self.input_layernorm(hidden), position_ids)
        residual = hidden
        h = self.post_attention_layernorm(hidden)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return residual + h


class MLAModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, head_dim, latent_dim,
                 intermediate_size, max_position_embeddings=1024, eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            MLABlock(d_model, n_heads, head_dim, latent_dim, intermediate_size, eps)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.embed_tokens(input_ids)
        for block in self.blocks:
            h = block(h, position_ids)
        h = self.norm(h)
        return self.lm_head(h)
