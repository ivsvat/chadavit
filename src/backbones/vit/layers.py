"""
ChAda-ViT (i.e Channel Adaptive ViT) is a variant of ViT that can handle multi-channel images.
"""
import math
from functools import partial
from typing import Optional, Union, Callable

import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

import torch
if torch.__version__=='2.5.1':
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from xformers.ops import memory_efficient_attention
    ATTN_DICT = {
        'torch.math' : SDPBackend.MATH,
        'torch.flash' : SDPBackend.FLASH_ATTENTION,
        'torch.efficient' : SDPBackend.EFFICIENT_ATTENTION,
        'torch.cudnn' : SDPBackend.CUDNN_ATTENTION,
    }
else:
    ATTN_DICT = {}


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _build_attention(
        embed_dim: int,
        num_heads: int,
        dropout: float,
        batch_first: bool,
        attn_type: str,
        **attn_kwargs,) -> nn.Module:
    
    if attn_type.startswith('torch'):
        return MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=batch_first, **attn_kwargs)
    
    elif attn_type.startswith('xFormers'):

        mha = xFormersMHSA(op=memory_efficient_attention)
        return mha
    else: raise NotImplementedError


class xFormersMHSA(Module):
    r"""
    Args:
        op (Callable): a function from xformers.ops
    """
    def __init__(
            self,
            op: Callable) -> None:
        super().__init__()
        self.op = op
    
    def forward(self, x):
        raise NotImplementedError
    


class TransformerEncoderLayer(Module):
    r"""
    Mostly copied from torch.nn.TransformerEncoderLayer, but with the following changes:
    - Added the possibility to retrieve the attention weights
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, return_attention = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], return_attention: bool = False) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=return_attention,
                            average_attn_weights=False)
        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TokenLearner(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class DevTransformerEncoderLayer(Module):
    r"""
    Mostly copied from torch.nn.TransformerEncoderLayer, but with the following changes:
    - Added the possibility to retrieve the attention weights
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None, attn_type: str='torch') -> None:
        super(DevTransformerEncoderLayer, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.attn_type = str(attn_type)
        _attn = _build_attention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first, attn_type=self.attn_type,
                                            **factory_kwargs)
        self.self_attn = _attn
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, return_attention = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], return_attention: bool = False) -> Tensor:
        if self.attn_type is not None and self.attn_type in ATTN_DICT.keys():
            with sdpa_kernel(ATTN_DICT[self.attn_type]):
                x, attn_weights = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=return_attention,
                                    average_attn_weights=False)
        else:
            x, attn_weights = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=return_attention,
                                average_attn_weights=False)

        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TokenLearner(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
