import torch
import torch.nn as nn
from einops import rearrange


class GroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        """
        We use a custom implementation for GroupNorm, since h=w=1 may raise some problem,
        see https://github.com/pytorch/pytorch/issues/115940
        """
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h, w = x.shape

        x = rearrange(x, "b (g n) h w -> b g (n h w)", g=self.num_groups)
        mean = torch.mean(x, dim=2, keepdim=True)
        variance = torch.var(x, dim=2, keepdim=True)

        x = (x - mean) / (variance + self.eps).sqrt()

        x = rearrange(x, "b g (n h w) -> b (g n) h w", h=h, w=w)

        x = x * self.weight + self.bias

        return x

    def extra_repr(self) -> str:
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"
