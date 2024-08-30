import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vector_quants.utils import VECTOR_QUANTS_DEBUG, print_module, print_params


class GMlpUnit(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        bias: bool = False,
        causal: bool = False,
        init_std: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        if VECTOR_QUANTS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        shape = num_heads, seq_len, seq_len
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        nn.init.normal_(self.weight, -init_std, init_std)

        self.causal = causal
        self.mask = torch.empty(0)

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
        **kwargs,
    ):
        # x: b n d
        b, n, d = x.shape

        # linear map
        u = self.in_proj(x)
        gate = F.silu(self.gate_proj(x))

        u = rearrange(u, "b n (h d) -> b h n d", h=self.num_heads)

        weight = self.weight
        if self.causal:
            if self.mask.size(0) == 0:
                mask = (1 - torch.ones(n, n, device=x.device)).bool()
                self.mask = mask
            weight = weight.masked_fill(self.mask, 0)

        output = torch.einsum("h n m, b h m d -> b h n d", weight, u)
        output = rearrange(output, "b h n d -> b n (h d)")

        output = gate * output

        # outproj
        output = self.out_proj(output)

        return output
