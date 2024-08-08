# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vision Transformers main function. This code is derived from ``DINO`` and ``timm`` library.
- ``DINO`` library: https://github.com/facebookresearch/dino (Apache License 2.0)
- ``timm`` library: https://github.com/rwightman/pytorch-image-models/blob/master/timm/ (Apache License 2.0)
"""

from typing import Callable, Dict, List, Optional, Tuple
from functools import partial

import math
import torch
from torch import nn

from .weight_init import trunc_normal_


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks). See [1]_ for details.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.
    drop_prob : float = 0.0
        The probability of dropping a path.
    training : bool = False
        Whether the model is in training mode or not.

    Returns
    -------
    torch.Tensor
        The output array after applying drop path.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L137
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """``torch.nn`` module for Drop path implementation. See [1]_ for details.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py#L157
    """

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """Initialize the DropPath class.

        Parameters
        ----------
        drop_prob : float
            The probability of dropping each element, by default None.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path to an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            Input tensor with dropout applied on a given path.
        """
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer. See [1]_ for details.

    Parameters
    ----------
    in_features : int
        Input feature size.
    hidden_features : int = None
        Hidden feature size, by default None (uses input feature size).
    out_features : int = None
        Output feature size, by default None (uses input feature size).
    act_layer : nn.Module = nn.GELU
        Activation layer, by default nn.GELU.
    drop : float = 0.0
        Dropout rate, by default 0.0.

    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer.
    act : nn.Module
        Activation layer.
    fc2 : nn.Linear
        Second fully connected layer.
    drop : nn.Dropout
        Dropout layer.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13

    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention module for Vision Transformer as implemented in [1]_.

    Parameters
    ----------
    dim : int
        Input dimension.
    num_heads : int = 8
        Number of attention heads, by default 8.
    qkv_bias : bool = False
        Whether to include biases in the linear transformations of q, k, and v,
        by default False.
    qk_scale : float = None
        Scale factor for qk dot product, by default None.
    attn_drop : float = 0.0
        Dropout rate for attention weights, by default 0.0.
    proj_drop : float = 0.0
        Dropout rate for output tensor, by default 0.0.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    scale : float
        Scale factor for qk dot product.
    qkv : nn.Linear
        Linear transformation for q, k, and v.
    attn_drop : nn.Dropout
        Dropout layer for attention weights.
    proj : nn.Linear
        Linear transformation for output tensor.
    proj_drop : nn.Dropout
        Dropout layer for output tensor.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the attention module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output tensor.
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Vision Transformer Block as implemented in [1]_.

    Parameters
    ----------
    dim : int
        The dimension of the input.
    num_heads : int
        The number of attention heads.
    mlp_ratio : float = 4.0
        The ratio of hidden dimension to input dimension in the MLP, by default 4.0.
    qkv_bias : bool = False
        Whether to include biases in the query, key, and value projections, by default False.
    qk_scale : float = None
        The scaling factor for query-key dot product, by default None.
    drop : float = 0.0
        The dropout probability, by default 0.0.
    attn_drop : float = 0.0
        The dropout probability for attention weights, by default 0.0.
    drop_path : float = 0.0
        The dropout probability for the residual connection, by default 0.0.
    act_layer : torch.nn.Module = torch.nn.GELU
        The activation layer used in the MLP, by default ``torch.nn.GELU``.
    norm_layer : torch.nn.Module = torch.nn.LayerNorm
        The normalization layer, by default ``torch.nn.LayerNorm``.
    init_values : int
        The initial values for gamma_1 and gamma_2, by default 0.

    Attributes
    ----------
    norm1 : torch.nn.Module
        The normalization layer before the attention module.
    attn : Attention
        The attention module.
    drop_path : Union[DropPath, nn.Identity]
        The dropout layer for the residual connection.
    norm2 : torch.nn.Module
        The normalization layer after the MLP.
    mlp : Mlp
        The MLP module.
    gamma_1 : Optional[torch.nn.Parameter]
        The learnable parameter gamma_1.
    gamma_2 : Optional[torch.nn.Parameter]
        The learnable parameter gamma_2.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,  # pylint: disable=redefined-outer-name
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        init_values: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer Block module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        return_attention : bool
            Whether to return the attention weights, by default False.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding as implemented in [1]_.

    Parameters
    ----------
    img_size:  int = 224
        Size of the input image (both height and width), by default 224.
    patch_size: int = 16
        Size of each patch, by default 16.
    in_chans: int = 3
        Number of input channels, by default 3.
    embed_dim: int = 768
        Dimension of the embedded patch representation, by default 768.

    Attributes
    ----------
    img_size: int
        Size of the input image (both height and width).
    patch_size: int
        Size of each patch.
    num_patches: int
        Total number of patches in the image.
    proj: nn.Conv2d
        Convolutional layer used for projection.


    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L25
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PatchEmbed module.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (``B``, ``C``, ``H``, ``W``) where ``B`` is the
            batch size, ``C`` is the number of channels, ``H`` is the input image
            height, and ``W`` is the input image width.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (``B``, ``embed_dim``, ``num_patches``) where
            ``B`` is the batch size, ``embed_dim`` is the dimension of the embedded
            patch  representation, and ``num_patches`` is the total number of
            patches in the image.
        """
        #  B, C, H, W = x.shape
        return self.proj(x)


class VisionTransformer(nn.Module):
    """Vision Transformer in PyTorch as implemented in [1]_.

    Parameters
    ----------
    img_size : List[int] = [224]
        The size of the input image.
    patch_size : int = 16
        The size of each patch in the input image.
    in_chans : int = 3
        The number of input channels.
    num_classes : int = 0
        The number of output classes.
    embed_dim : int = 768
        The dimension of the token embeddings.
    depth : int = 12
        The number of transformer blocks.
    num_heads : int = 12
        The number of attention heads.
    mlp_ratio : float = 4.0
        The ratio of the hidden dimension to the input dimension in the feed-forward network.
    qkv_bias : bool = False
        Whether to include biases to the query, key, and value linear layers.
    qk_scale : Optional[float] = None
        The scale factor for query and key.
    drop_rate : float = 0.0
        The dropout rate.
    attn_drop_rate : float = 0.0
        The dropout rate for attention probabilities.
    drop_path_rate : float = 0.0
        The dropout rate for residual connections.
    norm_layer : Callable = partial(torch.nn.LayerNorm, eps=1e-6)
        The normalization layer to be used.
    return_all_tokens : bool = False
        Whether to return all tokens or just the first token.
    init_values : int = 0
        The initial values for convolutional layers.
    use_mean_pooling : bool = False
        Whether to use mean pooling or not.
    masked_im_modeling : bool = False
        Whether to perform masked image modeling or not.

    References
    ----------
    .. [1] https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        img_size: List[int] = [224],
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        return_all_tokens: bool = False,
        init_values: int = 0,
        use_mean_pooling: bool = False,
        masked_im_modeling: bool = False,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        self.norm = (
            nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        )
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # Classifier head.
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # Masked image modeling.
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize the weights of the module.

        Parameters
        ----------
        m: torch.nn.Module
            Module to initialize weights for.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(
        self, x: torch.Tensor, w: int, h: int
    ) -> torch.Tensor:
        """Interpolate the positional encoding to match the size of the input
        tokens.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_tokens, embed_dim)
        w: int
            Width of the input image.
        h: int
            Height of the input image.

        Returns
        -------
        torch.Tensor
            Interpolated positional encoding tensor.
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat(
            (class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1
        )

    def prepare_tokens(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare the input tokens for the Vision Transformer.

        Parameters
        -----------
            x: torch.Tensor
                Input tensor of shape (batch_size, num_channels, width, height).
            mask: Optional[torch.Tensor] = None
                Mask tensor of shape (batch_size, width, height) for masked
                image modeling.

        Returns
        -------
        torch.Tensor
            Prepared tokens tensor.
        """
        B, nc, w, h = x.shape  # pylint: disable=unused-variable
        # patch linear embedding
        x = self.patch_embed(x)

        # mask image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: Optional[bool] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the Vision Transformer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_channels, width, height).
        return_all_tokens: Optional[bool] = None
            Whether to return the embeddings of all tokens.
        mask: Optional[torch.Tensor] = None
            Mask tensor of shape (batch_size, width, height) for masked image
            modeling.

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, embed_dim).

        """
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(  # pylint: disable=not-callable
                x[:, 1:, :].mean(1)
            )

        return_all_tokens = (
            self.return_all_tokens
            if return_all_tokens is None
            else return_all_tokens
        )
        if return_all_tokens:
            return x
        return x[:, 0]

    def extract_feature_maps(
        self, x: torch.Tensor, output_layers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Extract feature maps from the given input tensor.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, C, H, W).
        output_layers: List[str]
            List of output layer names.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing extracted feature maps for each output layer.
        """
        out_indices = [int(layer[5:]) for layer in output_layers]
        B, C, H, W = x.shape  # pylint: disable=unused-variable
        x = self.prepare_tokens(x)
        Hp = H // self.patch_embed.patch_size
        Wp = W // self.patch_embed.patch_size
        features = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                features["block" + str(i)] = xp
        return features

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Return the self-attention of the last block.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Self-attention tensor.
        """
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # Return attention of the last block.
                return blk(x, return_attention=True)

    def get_intermediate_layers(
        self, x: torch.Tensor, n: int = 1
    ) -> List[torch.Tensor]:
        """Return the output tokens from the last ``n`` last blocks.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        n: int = 1
            Number of last blocks to extract output from.

        Returns
        -------
        List[np.ndarray]
            List of output tokens from the ``n`` last blocks.
        """
        x = self.prepare_tokens(x)
        # Return the output tokens from the ``n`` last blocks.
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_num_layers(self) -> int:
        """Return the number of blocks (layers) in the model.

        Returns
        -------
        int
            Number of blocks.
        """
        return len(self.blocks)

    def mask_model(self, x: torch.Tensor, mask: torch.BoolTensor):
        """Apply a mask to the input tensor.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        mask: torch.BoolTensor
            Mask tensor.

        Returns
        -------
        torch.Tensor
            Masked tensor.
        """
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    """Create a Small Vision Transformer model (ViT-S).

    Parameters
    ----------
    patch_size : int = 16
        Size of the patches in the input image.
    **kwargs : keyword arguments
        Additional arguments to be passed to the VisionTransformer constructor.

    Returns
    -------
    model : VisionTransformer
        The small Vision Transformer model.
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def vit_base(patch_size: int = 16, **kwargs):
    """Create a Base Vision Transformer model (ViT-B).

    Parameters
    ----------
    patch_size : int = 16
        Size of the patches in the input image.
    **kwargs : keyword arguments
        Additional arguments to be passed to the VisionTransformer constructor.

    Returns
    -------
    model : VisionTransformer
        The Base Vision Transformer model.
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


def vit_large(patch_size: int = 16, **kwargs):
    """Create a Large Vision Transformer model (ViT-L).

    Parameters
    ----------
    patch_size : int = 16
        Size of the patches in the input image.
    **kwargs : keyword arguments
        Additional arguments to be passed to the VisionTransformer constructor.

    Returns
    -------
    model : VisionTransformer
        The Large Vision Transformer model.
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model
