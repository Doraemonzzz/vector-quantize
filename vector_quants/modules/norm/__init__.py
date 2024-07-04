import torch.nn as nn

from .srmsnorm import SRmsNorm

AUTO_NORM_MAPPING = {
    "layernorm": nn.LayerNorm,
    "srmsnorm": SRmsNorm,
}
