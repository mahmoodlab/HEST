import os
from pathlib import Path
import pdb
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

# Imports called in timm's vision_transformer.py
from timm.layers import Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn#, SwiGLUPacked
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv

# Imports of other functions called within timm's vision_transformer.py (that are not defined below)
from timm.models.vision_transformer import init_weights_vit_timm, get_init_weights_vit, _load_weights, _convert_openai_clip

# layers that that were adapated from timm
from .layers_latest import PatchEmbed

# see deprecation notice below:
def resize_pos_embed(model, pos_embed_w):
    resized = False
    if pos_embed_w.shape != model.pos_embed.shape:
        # see https://github.com/rwightman/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L509
        interpolation = 'bilinear'
        antialias = False
        try:
            from timm.layers import resample_abs_pos_embed
        except:
            print(f'{__file__}: import timm utility functions failed with version {timm.__version__}!')
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                    pos_embed_w,
                    new_size=model.patch_embed.grid_size,
                    num_prefix_tokens=num_prefix_tokens,
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        
        resized = True
    if not resized:
        logging.info('pos embedding not resized.')
    return pos_embed_w

### from timm==0.9.2
def resize_pos_embed_timm(
        posemb,
        posemb_new,
        num_prefix_tokens=1,
        gs_new=(),
        interpolation='bicubic',
        antialias=False,
):
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    #_logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

### No Major Change from 0.8.13dev0 -> 9.2.0
# fast attn -> fused attn naming
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.fast_attn = self.fused_attn # legacy support

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn and (return_attention == False):
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

### No Change from 0.8.13dev0 -> 9.2.0
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

### No Major Change from 0.8.13dev0 -> 9.2.0
# drop renamed as proj_drop
# Mlp layer added
class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0., # proj -> proj_drop renamed
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer( ### Mlp -> mlp_layer
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
    def forward_with_attention(self, x):
        x_input = x
        x_postattn, attn = self.attn(self.norm1(x_input), return_attention=True)
        x_postls1 = x_input + self.drop_path1(self.ls1(x_postattn))
        x_postmlp = self.mlp(self.norm2(x_postls1))
        x_postls2 = x_postls1 + self.drop_path2(self.ls2(x_postmlp))
        return x_postls2, attn


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Adapted entirely from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Last updated version: 0.8.17dev0
    """

    def __init__(
            self,
             img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 0,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,   # new
            patch_drop_rate: float = 0., # new
            proj_drop_rate: float = 0.,  # renamed
            attn_drop_rate: float = 0.,  # same
            drop_path_rate: float = 0.,  # same
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block,     
            mlp_layer: Callable = Mlp,  # New
            return_all_tokens=False,    # iBOT
            masked_im_modeling=False    # iBOT
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.return_all_tokens = return_all_tokens     # from ibot
        self.masked_im_modeling = masked_im_modeling   # from ibot

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            masked_im_modeling=masked_im_modeling,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        # new
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate, # renamed
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,  # new
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate) # new
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x, w, h):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.interpolate_pos_encoding(x, w, h)
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    ### all new
    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_class_token: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
    
    def forward_features(self, x, mask=None):
        B, nc, w, h = x.shape
        
        ### Specific to iBOT
        if self.masked_im_modeling:
            assert mask is not None
            x = self.patch_embed(x, mask) if self.masked_im_modeling else self.patch_embed(x)
        else:
            x = self.patch_embed(x)
        
        x = self._pos_embed(x, w, h)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x) # new
        return x if pre_logits else self.head(x)
    
    def get_attention(self, x, block_num: int=-1):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        x = self._pos_embed(x, w, h)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if block_num < 0:
            block_num = len(self.blocks) + block_num
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            for i, blk in enumerate(self.blocks):
                if i < block_num:
                    x = blk(x)
                else:
                    x, attn = blk.forward_with_attention(x)
                    return attn

    def forward(self, x, return_all_tokens=None, mask=None):
        if self.masked_im_modeling:
            assert mask is not None
            x = self.forward_features(x, mask)
        else:
            x = self.forward_features(x)
        
        ### iBOT
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        
        x = self.forward_head(x)
        return x
    
    
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def vit_latest_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

def vit_latest_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

def vit_latest_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

def vit_latest_base_clip_pretrained(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        pre_norm=True, norm_layer=nn.LayerNorm, # clip
        **kwargs)
    state_dict = torch.load(os.path.join('/'.join(str(Path(__file__).absolute()).split('/')[:-1]), 'vit_base_patch16_clip_224.openai.bin'), map_location="cpu")
    new_state_dict = _convert_openai_clip(state_dict, model)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print('Missing Keys:', missing)
    print('Unexpected Keys:', unexpected)
    return model

