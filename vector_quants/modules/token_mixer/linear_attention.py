import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vector_quants.utils import VECTOR_QUANTS_DEBUG, print_params

from ..norm import AUTO_NORM_MAPPING
from ..pe import MdLrpe


class LinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        use_lrpe: bool = False,
        lrpe_type: int = 1,
        base: int = 10000,
        causal: bool = False,
        norm_type: str = "layernorm",
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
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.output_gate = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim),
            nn.Linear(self.head_dim, embed_dim),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.norm = AUTO_NORM_MAPPING[norm_type](embed_dim)

        self.use_lrpe = use_lrpe
        self.causal = causal

        if self.use_lrpe:
            self.lrpe = MdLrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
            )

        self.mask = torch.empty(0)

    def forward(
        self,
        x,
        shape=None,
        **kwargs,
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        output_gate = F.sigmoid(self.output_gate(x))

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", d=self.head_dim),
            [q, k, v],
        )
        q = F.silu(q)
        k = F.silu(k)

        # lrpe
        if self.use_lrpe:
            q = self.lrpe(q, shape=shape)
            k = self.lrpe(k, shape=shape)

        if self.causal:
            if self.mask.size(0) == 0:
                self.mask = torch.tril(torch.ones(n, n, device=x.device))
            energy = torch.einsum("... h n d, ... h m d -> ... h n m", q, k) * self.mask
            output = torch.matmul(energy, v)
        else:
            kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
            output = torch.matmul(q, kv)

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")
        output = output * output_gate
        output = self.norm(output)

        # outproj
        output = self.out_proj(output)

        return output
