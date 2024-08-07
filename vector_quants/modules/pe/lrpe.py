"""
Lrpe in https://openreview.net/forum?id=xoLyps2qWc
"""

import torch
import torch.nn as nn
from einops import pack

from vector_quants.utils import VECTOR_QUANTS_DEBUG, logging_info, print_params


class MdLrpe(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 8,
        lrpe_type: int = 1,
        base: int = 10000,
    ):
        """
        lrpe_type: 1 for standard rope, 2 for mix rope (rope half hea dim), 3 for complex version (cosformer style)
        """
        super().__init__()
        if VECTOR_QUANTS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.lrpe_type = lrpe_type
        self.base = base
        self.theta_ = torch.empty(0)

        d = self.num_heads * self.head_dim
        if self.lrpe_type == 1:
            logging_info("standard rope")

            theta = base ** (
                -2 / d * torch.arange(d // 2, dtype=torch.int64)
            ).float().reshape(num_heads, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 2:
            logging_info("mix rope")
            assert head_dim % 2 == 0
            theta = base ** (
                -2 / d * torch.arange(d // 2 // 2, dtype=torch.int64)
            ).float().reshape(num_heads, -1)
            self.register_buffer("theta", theta, persistent=False)
        elif lrpe_type == 3:
            logging_info("complex transform")
            theta = base ** (
                -2 / d * torch.arange(d, dtype=torch.int64)
            ).float().reshape(num_heads, -1)
            self.register_buffer("theta", theta, persistent=False)
        else:
            raise ValueError(f"lrpe_type: {lrpe_type} has not been support!")

    def get_theta(self, x, shape=None):
        # x: b, h, ... , d
        # compute index
        if shape is None:
            shape = x.shape[2:-1]
        m = len(shape)
        array = [
            torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
            for n in shape
        ]
        theta = self.theta
        # h, d -> h, ..., d
        for _ in range(m):
            theta = theta.unsqueeze(1)
        grid = torch.meshgrid(array)
        index = torch.stack(grid, dim=-1)
        # compute theta
        if self.lrpe_type == 1:
            d = self.head_dim // 2 // m
        elif self.lrpe_type == 2:
            d = self.head_dim // 4 // m
        else:
            d = self.head_dim // m

        theta_ = []
        for i in range(m):
            theta_.append(index[..., i : i + 1] * theta[..., :d])

        theta_ = torch.cat(theta_, dim=-1)

        if len(x.shape) == 4:
            # b, h, n, d case
            theta_, ps = pack([theta_], "h * d")

        self.theta_ = theta_

    def extra_repr(self):
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, lrpe_type={self.lrpe_type}"

    def get_theta(self, offset=0):
        if offset == 0:
            return self.theta_
        else:
            return self.theta_[:, offset : offset + 1]

    def forward(self, x, shape=None, offset=0):
        n, d = x.shape[-2], x.shape[-1]
        if self.theta_.shape[0] == 0:
            self.get_theta(x, shape=shape)

        if offset > 0:
            assert len(shape) == 1, "current only support 1d lrpe for inference"

        if self.lrpe_type == 1:
            theta = self.get_theta(offset=offset)
            theta_ = torch.polar(
                torch.ones_like(theta).to(torch.float32), theta.to(torch.float32)
            )
            x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

            x_out = torch.view_as_real(x_ * theta_).flatten(3).type_as(x)
        elif self.lrpe_type == 2:
            e = d // 2
            # last e features
            x1 = x[..., e:]
            # do rope for the first e features
            x = x[..., :e]

            theta = self.get_theta(offset=offset)
            theta_ = torch.polar(
                torch.ones_like(theta).to(torch.float32), theta.to(torch.float32)
            )
            x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

            x_out = torch.cat(
                [torch.view_as_real(x_ * theta_).flatten(3).type_as(x), x1], dim=-1
            )

        elif self.lrpe_type == 3:
            theta = self.get_theta(offset=offset).float()

            x_out = torch.concat(
                [x * torch.cos(theta), x * torch.sin(theta)], dim=-1
            ).type_as(x)

        return x_out
