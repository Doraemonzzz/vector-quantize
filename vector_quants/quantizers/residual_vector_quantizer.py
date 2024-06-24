import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class ResidualVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self, num_embed, embed_dim, num_residual=1, commitment_loss_weight=0.25
    ):
        super().__init__()
        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.num_residual = num_residual
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
        indice_list = []
        loss_list = []
        x_quant = torch.zeros_like(x)
        residual = x.detach().clone()
        diff = 0

        for _ in range(self.num_residual):
            # get indice
            indice = self.latent_to_indice(residual)

            # quantize
            residual_quant = self.indice_to_code(indice)

            # compute diff
            loss_list.append(
                F.mse_loss(residual_quant, x.detach())
                + self.commitment_loss_weight * F.mse_loss(residual_quant.detach(), x)
            )

            # update
            residual = residual - residual_quant
            x_quant = x_quant + residual_quant
            indice_list.append(indice.unsqueeze(0))

        diff = torch.mean(torch.stack(loss_list))

        x_quant = x + (x_quant - x).detach()

        indice = torch.cat(indice_list, dim=0)

        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # n, m
        dist = compute_dist(latent, self.codebook.weight)
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice):
        return self.codebook(indice)
