import torch
import torch.nn.functional as F

AUTO_ACTIVATION_FN_MAPPING = {
    "gelu": F.gelu,
    "relu": F.relu,
    "elu": F.elu,
    "sigmoid": F.sigmoid,
    "exp": torch.exp,
    "leak": F.leaky_relu,
    "1+elu": lambda x: 1 + F.elu(x),
    "silu": F.silu,
}
