"""
GLU in https://arxiv.org/pdf/2002.05202.pdf
"""

import torch.nn as nn

from vector_quants.utils import VECTOR_QUANTS_DEBUG, print_params

from ..activations import AUTO_ACTIVATION_FN_MAPPING


class GLU(nn.Module):
    def __init__(
        self, embed_dim: int, mid_dim: int, act_fun: str, bias: bool = False
    ) -> None:
        super().__init__()

        if VECTOR_QUANTS_DEBUG:
            # get local varables
            params = locals()
            # print params
            print_params(**params)

        self.w1 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w2 = nn.Linear(embed_dim, mid_dim, bias=bias)
        self.w3 = nn.Linear(mid_dim, embed_dim, bias=bias)
        self.act = AUTO_ACTIVATION_FN_MAPPING[act_fun]

    def forward(self, x):
        output = self.w3(self.act(self.w1(x)) * self.w2(x))

        return output
