"""
ChAda-ViT (i.e Channel Adaptive ViT) is a variant of ViT that can handle multi-channel images.
"""

import math
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn

from torch import Tensor
import torch.nn.functional as F

from src.backbones.vit.dev_layers import DevTokenLearner
from src.backbones.vit.dev_layers import DevTransformerEncoderLayer
from src.utils.misc import trunc_normal_


class DevChAdaViT(nn.Module):
    """Channel Adaptive Vision Transformer"""

    def __init__(
        self,
        img_size=[224],
        in_chans=1,
        embed_dim=192,
        patch_size=16,
        num_classes=0,
        depth=12,
        num_heads=12,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        return_all_tokens=True,
        max_number_channels=10,
        **transformer_layer_kwargs,
    ):
        super().__init__()

        # Embeddings dimension
        self.num_features = self.embed_dim = embed_dim

        # Num of maximum channels in the batch
        self.max_channels = max_number_channels

        # Tokenization module
        self.token_learner = DevTokenLearner(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
        )
        num_patches = self.token_learner.num_patches

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim)
        )  # (B, max_channels * num_tokens, embed_dim)
        self.channel_token = nn.Parameter(
            torch.zeros(1, self.max_channels, 1, self.embed_dim)
        )  # (B, max_channels, 1, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, num_patches + 1, self.embed_dim)
        )  # (B, max_channels, num_tokens, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # TransformerEncoder block
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                DevTransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=num_heads,
                    dim_feedforward=2048,
                    dropout=dpr[i],
                    batch_first=True,
                    **transformer_layer_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Return only the [CLS] token or all tokens
        self.return_all_tokens = return_all_tokens

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.channel_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_pos_encoding_per_channel(self, x, w, h, class_pos_embed: bool = False):
        """
        Adds num_patches positional embeddings to EACH of the channels.
        """
        npatch = x.shape[2]
        N = self.pos_embed.shape[2] - 1

        # --------------------- [CLS] positional encoding --------------------- #
        if class_pos_embed:
            return self.pos_embed[:, :, 0]

        # --------------------- Patches positional encoding --------------------- #
        # If the input size is the same as the training size, return the positional embeddings for the desired type
        if npatch == N and w == h:
            return self.pos_embed[:, :, 1:]

        # Otherwise, interpolate the positional encoding for the input tokens
        class_pos_embed = self.pos_embed[:, :, 0]
        patch_pos_embed = self.pos_embed[:, :, 1:]
        dim = x.shape[-1]
        w0 = w // self.token_learner.patch_size
        h0 = h // self.token_learner.patch_size
        # a small number is added by DINO team to avoid floating point error in the interpolation
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
        return patch_pos_embed.unsqueeze(0)

    def channel_aware_tokenization(self, x, index, list_num_channels, max_channels=10):
        B, nc, w, h = x.shape  # (B*num_channels, 1, w, h)

        # Tokenize through linear embedding
        tokens_per_channel = self.token_learner(x) # (sum(list_num_channels), n_patches=(224/16)**2, embedding_dim)

        # Concatenate tokens per channel in each image
        chunks = torch.split(
            tokens_per_channel, list_num_channels[index], dim=0
        )  # List of (img_channels, n_patches, embed_dim)

        # Pad the tokens tensor with zeros for each image separately in the chunks list
        padded_tokens = [
            (
                torch.cat(
                    [
                        chunk,
                        torch.zeros(
                            (
                                max_channels - chunk.size(0),
                                chunk.size(1),
                                chunk.size(2),
                            ),
                            device=chunk.device,
                        ),
                    ],
                    dim=0,
                )
                if chunk.size(0) < max_channels
                else chunk
            )
            for chunk in chunks
        ]

        # Stack along the batch dimension
        padded_tokens = torch.stack(
            padded_tokens, dim=0
        )  # (B, img_channels, num_tokens, embed_dim)
        num_tokens = padded_tokens.size(2)

        # Reshape the patches embeddings on the channel dimension
        padded_tokens = padded_tokens.reshape(
            padded_tokens.size(0), -1, padded_tokens.size(3)
        )  # (B, max_channels*num_tokens, embed_dim)

        # Compute the masking for avoiding self-attention on empty padded channels => CRUCIAL to perform this operation here, before having added the [POS] and [CHANNEL] tokens
        channel_mask = torch.all(
            padded_tokens == 0.0, dim=-1
        )  # (B, max_channels*num_tokens) Check if all elements in the last dimension are zeros (indicating a padded channel)
    
        # Destack to obtain the original number of channels
        padded_tokens = padded_tokens.reshape(
            -1, max_channels, num_tokens, padded_tokens.size(-1)
        )  # (B, img_channels, num_tokens, embed_dim)

        # Add the [POS] token to the embed patch tokens
        padded_tokens = padded_tokens + self.add_pos_encoding_per_channel(
            padded_tokens, w, h, class_pos_embed=False
        )

        # Add the [CHANNEL] token to the embed patch tokens
        if max_channels == self.max_channels:
            channel_tokens = self.channel_token.expand(
                padded_tokens.shape[0], -1, padded_tokens.shape[2], -1
            )  # (1, max_channels, 1, embed_dim) -> (B, max_channels, num_tokens, embed_dim)
            padded_tokens = (
                padded_tokens + channel_tokens
            )  # Add a different channel_token to the ALL the patches of a same channel

        ########################### Sanity Check ###########################
        # self.channel_token_sanity_check(channel_tokens)

        # Restack the patches embeddings on the channel dimension
        embeddings = padded_tokens.reshape(
            padded_tokens.size(0), -1, padded_tokens.size(3)
        )  # (B, max_channels*num_tokens, embed_dim)

        # Expand the [CLS] token to the batch dimension
        cls_tokens = self.cls_token.expand(
            embeddings.shape[0], -1, -1
        )  # (1, 1, embed_dim) -> (B, 1, embed_dim)

        # Add [POS] positional encoding to the [CLS] token
        cls_tokens = cls_tokens + self.add_pos_encoding_per_channel(
            embeddings, w, h, class_pos_embed=True
        )

        # Concatenate the [CLS] token to the embed patch tokens
        embeddings = torch.cat(
            [cls_tokens, embeddings], dim=1
        )  # Append cls_token to the beginning of each image

        # Adding a False value to the beginning of each channel_mask to account for the [CLS] token
        channel_mask = torch.cat(
            [
                torch.tensor([False], device=channel_mask.device).expand(
                    channel_mask.size(0), 1
                ),
                channel_mask,
            ],
            dim=1,
        )

        return self.pos_drop(embeddings), channel_mask
    
    def nested_tokenization(self, x, index, list_num_channels, max_channels=10):
        r"""Tokenize input and form nested tensors

        NOTE: doesn't have POSITIONAL EMBEDDINGS

        Args: see self.channel_aware_tokenization
        Return:

        """
        B, nc, w, h = x.shape  # (B*num_channels, 1, w, h)

        # Tokenize through linear embedding
        tokens_per_channel = self.token_learner(x) # (sum(list_num_channels), n_patches=(224/16)**2, embedding_dim)

        # Concatenate tokens per channel in each image
        chunks = torch.split(
            tokens_per_channel, list_num_channels[index], dim=0
        )  # List of (img_channels, n_patches, embed_dim)
        flattented_chunks = [chunk.view(-1, self.embed_dim) for chunk in chunks]
        

    def forward(self, x, index, list_num_channels):
        # Apply the TokenLearner module to obtain learnable tokens
        x, channel_mask = self.channel_aware_tokenization(
            x, index, list_num_channels
        )  # (B*num_channels, embed_dim)

        # Apply the self-attention layers with masked self-attention
        for blk in self.blocks:
            x = blk(
                x, src_key_padding_mask=channel_mask
            )  # Use src_key_padding_mask to mask out padded tokens

        # Normalize
        x = self.norm(x)

        if self.return_all_tokens:
            # Create a mask to select non-masked tokens (excluding CLS token)
            non_masked_tokens_mask = ~channel_mask[:, 1:]
            non_masked_tokens = x[:, 1:][non_masked_tokens_mask]
            return non_masked_tokens  # return non-masked tokens (excluding CLS token)
        else:
            return x[:, 0]  # return only the [CLS] token

    def channel_token_sanity_check(self, x):
        """
        Helper function to check consistency of channel tokens.
        """
        # 1. Compare Patches Across Different Channels
        print("Values for the first patch across different channels:")
        for ch in range(10):  # Assuming 10 channels
            print(
                f"Channel {ch + 1}:", x[0, ch, 0, :5]
            )  # Print first 5 values of the embedding for brevity

        print("\n")

        # 2. Compare Patches Within the Same Channel
        for ch in range(10):
            is_same = torch.all(x[0, ch, 0] == x[0, ch, 1])
            print(
                f"First and second patch embeddings are the same for Channel {ch + 1}: {is_same.item()}"
            )

        # 3. Check Consistency Across Batch
        print("Checking consistency of channel tokens across the batch:")
        for ch in range(10):
            is_consistent = torch.all(x[0, ch, 0] == x[1, ch, 0])
            print(
                f"Channel token for first patch is consistent between first and second image for Channel {ch + 1}: {is_consistent.item()}"
            )

    def get_last_selfattention(self, x):
        x, channel_mask = self.channel_aware_tokenization(
            x, index=0, list_num_channels=[1], max_channels=1
        )
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, src_key_padding_mask=channel_mask)
            else:
                # return attention of the last block
                return blk(x, src_key_padding_mask=channel_mask, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x, channel_mask = self.channel_aware_tokenization(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, src_key_padding_mask=channel_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def dev_chada_vit(**kwargs):
    patch_size = kwargs.pop("patch_size")
    embed_dim = kwargs.pop("embed_dim")
    return_all_tokens = kwargs.pop("return_all_tokens")
    max_number_channels = kwargs.pop("max_number_channels")
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 2)
    kwargs.setdefault("norm_layer", partial(nn.LayerNorm, eps=1e-6))
    model = DevChAdaViT(
        patch_size=patch_size,
        embed_dim=embed_dim,
        return_all_tokens=return_all_tokens,
        max_number_channels=max_number_channels,
        **kwargs,
    )
    return model
