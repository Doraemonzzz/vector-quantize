import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_embed = cfg.num_embed
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        entropy_temperature = cfg.entropy_temperature
        entropy_loss_type = cfg.entropy_loss_type
        entropy_loss_weight = cfg.entropy_loss_weight
        # get params end

        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_temperature = entropy_temperature
        self.entropy_loss_type = entropy_loss_type
        self.entropy_loss_weight = entropy_loss_weight

        # init codebook
        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):

        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute codebook loss
        codebook_loss = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        entropy_loss = self.entropy_loss(x)

        loss_dict = {
            "codebook_loss": codebook_loss,
            "entropy_loss": entropy_loss,
        }

        x_quant = x + (x_quant - x).detach()

        return x_quant, indice, loss_dict

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
