import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vector_quants.utils import VECTOR_QUANTS_DEBUG, print_params

from ..pe import MdLrpe


class SoftmaxAttentionAr(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        use_lrpe: bool = False,
        lrpe_type: int = 1,
        base: int = 10000,
        causal: bool = False,
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

        if self.use_lrpe:
            self.lrpe = MdLrpe(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                lrpe_type=lrpe_type,
                base=base,
            )

    def forward(
        self,
        x,
        past_key_value=None,
        shape=None,
        **kwargs,
    ):
        # x: b n d

        # linear map
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", d=self.head_dim),
            [q, k, v],
        )

        q_offset = 0
        # lrpe relys on position, get cache first
        if past_key_value is not None:
            # reuse k, v, for evaluation only
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)
            q_offset = past_key_value[0].shape[-2]

        past_key_value = (k, v) if not self.training else None
        # lrpe
        if self.use_lrpe:
            q = self.lrpe(q, shape=shape, offset=q_offset)
            k = self.lrpe(k, shape=shape)
        print(self.causal)
        # during inference, since we use kv cache, the attention is not causal
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal if self.training else False
        )
        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        # outproj
        output = self.out_proj(output)

        return output, past_key_value
