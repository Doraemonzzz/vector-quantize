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
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        latent = rearrange(latent, "... (g d) -> (... g) d", g=self.num_group)
        # n, m
        dist = compute_dist(latent, self.codebook.weight)
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = rearrange(indice, "(b g) -> b g", g=self.num_group)
        indice = unpack_one(indice, ps, "* g")

        return indice

    def indice_to_code(self, indice):
        code = self.codebook(indice)
        code = rearrange(code, "... g d -> ... (g d)")

        return code
