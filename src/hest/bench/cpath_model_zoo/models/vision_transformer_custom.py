from .vision_transformer_latest import VisionTransformer, resize_pos_embed
from typing import Optional, List, Callable
import torch
import torch.nn as nn
from torch import einsum
import pdb
from einops import rearrange, repeat
from .attentional_poolers import AttentionalPooler, AttentionalPoolerLegacy
from einops_exts import rearrange_many, repeat_many

class EncoderWithAttentionalPooler(nn.Module):
    def __init__(
        self,
        encoder,
        attn_pooler_contrast,
        embed_dim,
        norm_layer: Callable = nn.LayerNorm,
        global_average_pool: bool = False
    ):
        super().__init__()
        self.trunk = encoder
        self.attn_pool_contrast = attn_pooler_contrast
        self.ln_contrast = norm_layer(embed_dim)
        self.global_average_pool = global_average_pool
    
    def _global_pool(self, x):
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]
    
    def forward(self, x):
        x = self.trunk(x, return_all_tokens=True)
        if self.global_average_pool:
            pooled, _ = self._global_pool(x)
        else:
            pooled = self.attn_pool_contrast(x)[:, 0]
            pooled = self.ln_contrast(pooled)
        return pooled
    
def vit_base_w_pooler(patch_size=16, pooler_n_queries_contrast=1, legacy=True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    if legacy:
        attn_pooler_contrast = AttentionalPoolerLegacy(d_model=512, 
                                             context_dim=768, 
                                             n_queries=pooler_n_queries_contrast)
    else:
        attn_pooler_contrast = AttentionalPooler(d_model=512, 
                                                context_dim=768, 
                                                n_queries=pooler_n_queries_contrast)
    model = EncoderWithAttentionalPooler(encoder=model, 
                                         attn_pooler_contrast=attn_pooler_contrast, 
                                         embed_dim=512)
    return model

def vit_large_w_pooler(patch_size=16, pooler_n_queries_contrast=1, legacy=True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, **kwargs)
    if legacy:
        attn_pooler_contrast = AttentionalPoolerLegacy(d_model=768, 
                                             context_dim=1024, 
                                             n_queries=pooler_n_queries_contrast)
    else:
        attn_pooler_contrast = AttentionalPooler(d_model=768, 
                                                context_dim=1024, 
                                                n_queries=pooler_n_queries_contrast)
    model = EncoderWithAttentionalPooler(encoder=model, 
                                         attn_pooler_contrast=attn_pooler_contrast, 
                                         embed_dim=768)
    return model