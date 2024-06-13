import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embed, embed_dim, commitment_loss_weight=0.25):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook()

    def init_codebook(self):
        nn.init.normal_(self.codebook.weight, mean=0, std=self.embed_dim**-0.5)

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
        # n, m
        dist = compute_dist(codes, self.codebook.weight)
        # n, 1
        indices = torch.argmin(dist, dim=-1)
        indices = unpack_one(indices, ps, "*")

        return indices

    def indices_to_codes(self, indices):
        return self.codebook(indices)
