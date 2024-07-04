import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vector_quants.utils import VECTOR_QUANTS_DEBUG, print_params

# Lrpe = None


class SoftmaxAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        use_lrpe: bool = True,
        lrpe_type: int = 1,
        base: int = 10000,
        causal=False,
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
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.use_lrpe = use_lrpe
        self.causal = causal

        # if self.use_lrpe:
        #     self.lrpe = Lrpe(
        #         head_dim=self.head_dim,
        #         num_heads=self.num_heads,
        #         lrpe_type=lrpe_type,
        #         base=base,
        #     )

    def forward(
        self,
        x,
        **kwargs,
    ):
        # x: b n d

        # linear map
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", d=self.head_dim),
            [q, k, v],
        )

        # # lrpe
        # if self.use_lrpe:
        #     q = self.lrpe(q)
        #     k = self.lrpe(k)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        # q, k, v = map(lambda x: rearrange(x, "... h n d -> ... n h d"), [q, k, v])

        # if attention_mask is None:
        #     output = flash_attn_func(q, k, v, causal=True)
        # else:
        #     assert False, "flash_attn_varlen_qkvpacked_func current not support"

        # # reshape
        # output = rearrange(output, "... n h d -> ... n (h d)")

        # outproj
        output = self.out_proj(output)

        return output
