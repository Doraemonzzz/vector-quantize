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
        # get params end

        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.kl_temperature = kl_temperature
        self.kl_loss_weight = kl_loss_weight

        # init codebook

        self.init_codebook()

        # v2
        # self.proj = nn.Conv2d(self.embed_dim, self.num_embed, 1)
        # v4
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

        # compute codebook loss
        # codebook_loss = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)
        loss_dict = {
            # "codebook_loss": codebook_loss,
            "kl_loss": kl_loss,
        }

        # x_quant = x + (x_quant - x).detach()

        indice = indice.argmax(dim=-1).detach()

        return x_quant, indice, loss_dict

    # v1
    # not converge
    # def latent_to_indice(self, latent, eps=1e-10):
    #     # (b, *, d) -> (n, d)
    #     latent, ps = pack_one(latent, "* d")
    #     # n, m
    #     dist = compute_dist(latent, self.codebook.weight)
    #     # n, m
    #     hard = False if self.training else True
    #     indice = F.gumbel_softmax(-dist, tau=self.kl_temperature, dim=-1, hard=hard)
    #     indice = unpack_one(indice, ps, "* m")

    #     kl_loss = self.kl_loss(latent=None, dist=dist)

    #     return indice, kl_loss

    # # v2: seem ok
    # def latent_to_indice(self, latent, eps=1e-10):
    #     latent = rearrange(latent, "b h w c -> b c h w")
    #     logits = self.proj(latent)
    #     # (b, *, d) -> (n, d)
    #     logits, ps = pack_one(logits, "* d")
    #     # n, m
    #     hard = False if self.training else True
    #     # indice = F.gumbel_softmax(-dist, tau=self.kl_temperature, dim=-1, hard=hard)
    #     indice = F.gumbel_softmax(logits, tau=self.kl_temperature, dim=-1, hard=hard)
    #     indice = unpack_one(indice, ps, "* m")
    #     indice = rearrange(indice, "b m h w -> b h w m")

    #     kl_loss = self.kl_loss(latent=None, dist=logits)

    #     return indice, kl_loss

    # # v3: seem not work
    # def latent_to_indice(self, latent, eps=1e-10):
    #     logits = torch.einsum("... d, m d -> ... m", latent, self.codebook.weight)
    #     # (b, *, d) -> (n, d)
    #     logits, ps = pack_one(logits, "* d")
    #     # n, m
    #     hard = False if self.training else True
    #     # indice = F.gumbel_softmax(-dist, tau=self.kl_temperature, dim=-1, hard=hard)
    #     indice = F.gumbel_softmax(logits, tau=self.kl_temperature, dim=-1, hard=hard)
    #     indice = unpack_one(indice, ps, "* m")

    #     kl_loss = self.kl_loss(latent=None, dist=logits)
    #     return indice, kl_loss

    # v4
    def latent_to_indice(self, latent, eps=1e-10):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        logits = self.proj(latent)
        # n, m
        hard = False if self.training else True
        indice = F.gumbel_softmax(logits, tau=self.kl_temperature, dim=-1, hard=hard)
        indice = unpack_one(indice, ps, "* m")

        kl_loss = self.kl_loss(latent=None, dist=logits)
        return indice, kl_loss

    def indice_to_code(self, indice):
        return torch.einsum("... m, m d -> ... d", indice, self.codebook.weight)
