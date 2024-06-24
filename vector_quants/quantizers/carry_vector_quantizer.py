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

    def __init__(self, base, num_levels, embed_dim, commitment_loss_weight=0.25):
        super().__init__()
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
        self.register_buffer("_offset", _levels.cumsum(dim=0), persistent=False)
        assert embed_dim % self.num_levels == 0
        self.embed_dim = embed_dim // self._levels.shape[0]
        self.commitment_loss_weight = commitment_loss_weight

        # create the codebook of the desired size
        self.codebook_weight = nn.Parameter(
            torch.empty(base, self.embed_dim), requires_grad=True
        )

        self.init_codebook()

    def extra_repr(self):
        return print_module(self)

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        nn.init.uniform_(self.codebook_weight, -1 / self.base, 1 / self.base)

    def forward(self, x):
        # get indices
        indices = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indices)

        # compute diff
        diff = F.mse_loss(
            x_quant, x.detach()
        ) + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indices

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # compute in parallel
        latent = rearrange(latent, "... (g d) -> (... g) d", g=self.num_levels)
        # n, m
        dist = compute_dist(latent, self.codebook_weight)
        # n, 1
        indices = torch.argmin(dist, dim=-1)
        indices = rearrange(indices, "(b g) -> b g", g=self.num_levels)
        indices = (indices * self._basis).sum(dim=-1).to(torch.int32)

        indices = unpack_one(indices, ps, "*")

        return indices

    def indice_to_code(self, indices):
        indices = (indices.unsqueeze(-1) // self._basis) % self._levels
        codes_list = []
        for i in range(self.num_levels):
            codes = F.embedding(indices[..., i], self.codebook_weight)
            codes_list.append(codes.unsqueeze(-1))
        codes = rearrange(torch.cat(codes_list, dim=-1), "... d g -> ... (g d)")

        return codes
