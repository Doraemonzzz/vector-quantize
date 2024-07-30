import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vector_quants.utils import print_module

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class CarryVectorQuantizer(BaseVectorQuantizer):
    """
    Similar to the Hierarchical Vector Quantizer, the only difference is that it uses the same level and shares the codebook.
    """

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        base = cfg.base
        num_levels = cfg.num_levels
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        # get params end

        self.base = base
        levels = [base] * num_levels
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]

        # new version
        if embed_dim % self.num_levels == 0:
            self.pad = 0
            self.embed_dim = (embed_dim + self.pad) // self.num_levels
        else:
            self.pad = self.num_levels - embed_dim % self.num_levels
            self.embed_dim = (embed_dim + self.pad) // self.num_levels

        self.commitment_loss_weight = commitment_loss_weight

        self.init_codebook()

    def extra_repr(self):
        return print_module(self)

    @property
    def num_embed(self):
        return self.base

    def init_codebook(self):
        self.codebook_weight = nn.Parameter(
            torch.empty(self.base, self.embed_dim), requires_grad=True
        )
        nn.init.uniform_(self.codebook_weight, -1 / self.base, 1 / self.base)

    def forward(self, x):
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute codebook loss
        codebook_loss = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        loss_dict = {
            "codebook_loss": codebook_loss,
        }

        x_quant = x + (x_quant - x).detach()

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        latent = F.pad(latent, (0, self.pad))
        # compute in parallel
        latent = rearrange(latent, "... (g d) -> (... g) d", g=self.num_levels)
        # n, m
        dist = compute_dist(latent, self.codebook_weight)
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = rearrange(indice, "(b g) -> b g", g=self.num_levels)

        indice = unpack_one(indice, ps, "* g")

        return indice

    def indice_to_code(self, indice):
        code_list = []
        for i in range(self.num_levels):
            code = F.embedding(indice[..., i], self.codebook_weight)
            code_list.append(code.unsqueeze(-1))
        code = rearrange(torch.cat(code_list, dim=-1), "... d g -> ... (g d)")

        if self.pad:
            code = code[..., : -self.pad]

        return code
