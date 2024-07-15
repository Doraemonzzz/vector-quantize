import torch
import torch.nn as nn
from einops import pack


class SinCosPe(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        base: int = 10000,
    ):
        super().__init__()

        theta = (
            base
            ** (
                -2 / embed_dim * torch.arange(embed_dim // 2, dtype=torch.int64)
            ).float()
        )
        self.register_buffer("theta", theta, persistent=False)
        self.embed_dim = embed_dim
        self.pe = torch.empty(0)

    def extra_repr(self):
        return f"embed_dim={self.embed_dim}"

    def get_pe(self, x, shape=None):
        # x: b, ... , d
        # compute index
        if shape is None:
            shape = x.shape[1:-1]
        m = len(shape)
        array = [
            torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
            for n in shape
        ]
        grid = torch.meshgrid(array)
        index = torch.stack(grid, dim=-1)

        # compute theta
        d = self.embed_dim // m

        theta = []
        for i in range(m):
            s = i * d
            e = min(s + d, self.embed_dim)
            theta.append(index[..., i : i + 1] * self.theta[s:e])

        theta = torch.cat(theta, dim=-1)

        # compute pe
        pe = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)

        if len(x.shape) == 3:
            # b, n, d case
            pe, ps = pack([pe], "* d")

        self.pe = pe

    def forward(self, x, shape=None):
        if self.pe.shape[0] == 0:
            self.get_pe(x, shape)
        x = x + self.pe

        return x
