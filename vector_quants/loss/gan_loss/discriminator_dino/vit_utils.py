# credit to: https://github.com/autonomousvision/stylegan-t

"""Flexible configuration and feature extraction of timm VisionTransformers."""

import math
import types
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddReadout(nn.Module):
    def __init__(self, start_index: bool = 1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        readout = x[:, : self.start_index].mean(dim=1)

        return x[:, self.start_index :] + readout.unsqueeze(1)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()


def forward_vit(pretrained: nn.Module, x: torch.Tensor) -> dict:
    _, _, H, W = x.size()
    _ = pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    if not self.use_reg:
        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index :],
        )
    else:
        posemb_grid = posemb[0]

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    if not self.use_reg:
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    else:
        posemb = posemb_grid

    return posemb


def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    # patch proj and dynamically resize
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(
        self.pos_embed, H // self.patch_size[1], W // self.patch_size[0]
    )

    if not self.use_reg:
        # add cls token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # forward pass
        x = x + pos_embed
    else:
        # forward pass
        x = x + pos_embed

        to_cat = []
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        x = torch.cat(to_cat + [x], dim=1)

    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)
    return x


activations = {}


def get_activation(name: str) -> Callable:
    def hook(model, input, output):
        activations[name] = output

    return hook


def make_vit_backbone(
    model: nn.Module,
    hooks: list[int] = [2, 5, 8, 11],
    hook_patch: bool = True,
    # start_index: list[int] = 1,
    use_reg: bool = False,
):
    assert len(hooks) == 4

    pretrained = nn.Module()
    pretrained.model = model

    # add hooks
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("0"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("3"))
    if hook_patch:
        pretrained.model.pos_drop.register_forward_hook(get_activation("4"))

    # configure readout
    if not use_reg:
        start_index = 1
    else:
        start_index = 1 + pretrained.model.reg_token.shape[1]

    pretrained.rearrange = nn.Sequential(AddReadout(start_index), Transpose(1, 2))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = model.patch_embed.patch_size
    pretrained.model.use_reg = use_reg

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained
