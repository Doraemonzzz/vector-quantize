import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class GroupVectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embed, embed_dim, num_group, commitment_loss_weight=0.25):
        super().__init__()
        self._num_embed = num_embed
        self.num_group = num_group
        self.embed_dim = embed_dim // num_group
        self.commitment_loss_weight = commitment_loss_weight

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        # nn.init.normal_(self.codebook.weight, mean=0, std=self.embed_dim**-0.5)
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):
        # get indices
        indices = self.codes_to_indices(x)

        # quantize
        x_quant = self.indices_to_codes(indices)

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indices

    def codes_to_indices(self, codes):
        # (b, *, d) -> (n, d)
        codes, ps = pack_one(codes, "* d")
        codes = rearrange(codes, "... (g d) -> (... g) d", g=self.num_group)
        # n, m
        dist = compute_dist(codes, self.codebook.weight)
        # n, 1
        indices = torch.argmin(dist, dim=-1)
        indices = rearrange(indices, "(b g) -> b g", g=self.num_group)
        indices = unpack_one(indices, ps, "* g")

        return indices

    def indices_to_codes(self, indices):
        codes = self.codebook(indices)
        codes = rearrange(codes, "... g d -> ... (g d)")

        return codes
