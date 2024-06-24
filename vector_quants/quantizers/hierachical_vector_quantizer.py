import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class HierachicalVectorQuantizer(BaseVectorQuantizer):
    def __init__(self, levels, embed_dim, commitment_loss_weight=0.25):
        super().__init__()

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
        self.codebook_weights = nn.ParameterList(
            nn.Parameter(torch.empty(n, self.embed_dim), requires_grad=True)
            for n in levels
        )
        self.init_codebook()

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        for i in range(self.num_levels):
            n = self._levels[i]
            nn.init.uniform_(self.codebook_weights[i], -1 / n, 1 / n)

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
        latent = rearrange(latent, "... (g d) -> ... d g", g=self.num_levels)
        indices_list = []
        for i in range(self.num_levels):
            # n, m
            dist = compute_dist(latent[..., i], self.codebook_weights[i])
            # n, 1
            indices = torch.argmin(dist, dim=-1)
            indices_list.append(indices.unsqueeze(-1))

        indices = (
            (torch.cat(indices_list, dim=-1) * self._basis).sum(dim=-1).to(torch.int32)
        )
        indices = unpack_one(indices, ps, "*")

        return indices

    def indice_to_code(self, indices):
        indices = (indices.unsqueeze(-1) // self._basis) % self._levels
        codes_list = []
        for i in range(self.num_levels):
            codes = F.embedding(indices[..., i], self.codebook_weights[i])
            codes_list.append(codes.unsqueeze(-1))
        codes = rearrange(torch.cat(codes_list, dim=-1), "... d g -> ... (g d)")

        return codes
