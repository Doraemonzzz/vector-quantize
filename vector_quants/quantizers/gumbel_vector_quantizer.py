import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import pack_one, unpack_one


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_embed = cfg.num_embed
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        kl_temperature = cfg.kl_temperature
        kl_loss_weight = cfg.kl_loss_weight
        bias = cfg.bias
        straight_through = cfg.straight_through
        # get params end

        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.kl_temperature = kl_temperature
        self.kl_loss_weight = kl_loss_weight
        self.straight_through = straight_through
        # init codebook

        self.init_codebook()

        self.proj = nn.Linear(self.embed_dim, self.num_embed, bias=bias)

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):
        # get indice
        indice, kl_loss = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        loss_dict = {
            "kl_loss": kl_loss,
        }

        indice = indice.argmax(dim=-1).detach()

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent, eps=1e-10):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        logits = self.proj(latent)
        # n, m
        hard = self.straight_through if self.training else True

        indice = F.gumbel_softmax(logits, tau=self.kl_temperature, dim=-1, hard=hard)
        indice = unpack_one(indice, ps, "* m")

        kl_loss = self.kl_loss(logits=logits)
        return indice, kl_loss

    def indice_to_code(self, indice):
        return torch.einsum("... m, m d -> ... d", indice, self.codebook.weight)
