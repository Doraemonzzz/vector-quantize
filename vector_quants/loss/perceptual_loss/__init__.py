# credit to https://github.com/S-aiueo32/lpips-pytorch

import torch

from .lpips import LPIPS
from .lpips_timm import LpipsTimm


def lpips(
    x: torch.Tensor, y: torch.Tensor, net_type: str = "alex", version: str = "0.1"
):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
