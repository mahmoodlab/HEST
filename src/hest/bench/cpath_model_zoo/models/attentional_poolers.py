from typing import Optional, List, Callable
import torch
import torch.nn as nn
from torch import einsum
import pdb
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        dim_head = d_model // n_head

        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head

        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        q = repeat(self.query, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        x = self.ln_k(x)
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(q)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out).squeeze(dim=1)


class AttentionalPoolerLegacy(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)