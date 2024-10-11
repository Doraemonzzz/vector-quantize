import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class GroupSepVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_embed = cfg.num_embed
        embed_dim = cfg.embed_dim
        num_group = cfg.num_group
        commitment_loss_weight = cfg.commitment_loss_weight
        vq_norm_type = cfg.vq_norm_type
        # get params end

        self._num_embed = num_embed
        self.num_group = num_group
        # add pad
        if embed_dim % num_group == 0:
            self.pad = 0
        else:
            self.pad = num_group - embed_dim % num_group
        self.embed_dim = (embed_dim + self.pad) // num_group
        self.commitment_loss_weight = commitment_loss_weight
        self.vq_norm_type = vq_norm_type
        if self.vq_norm_type == "l2":
            self.fn = lambda x: F.normalize(x, p=2, dim=-1)
        elif self.vq_norm_type == "l1":
            self.fn = lambda x: F.normalize(x, p=1, dim=-1)
        else:
            self.fn = lambda x: x

        levels = [num_embed] * num_group
        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("_basis", _basis, persistent=False)

        # init codebook
        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    def init_codebook(self):
        self.codebook = nn.Parameter(
            torch.empty(self.num_group, self.num_embed, self.embed_dim),
            requires_grad=True,
        )
        nn.init.uniform_(self.codebook, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute codebook loss
        commitment_loss = F.mse_loss(x_quant.detach(), x)
        codebook_loss = (
            F.mse_loss(x_quant, x.detach())
            + self.commitment_loss_weight * commitment_loss
        )
        loss_dict = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }

        x_quant = x + (x_quant - x).detach()

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent):
        latent = F.pad(latent, (0, self.pad))
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        latent = rearrange(latent, "... (g d) -> g ... d", g=self.num_group)
        latent = self.fn(latent)
        # g, n, m
        dist = compute_dist(latent, self.fn(self.codebook))
        # g, n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = rearrange(indice, "g ... -> ... g")
        indice = unpack_one(indice, ps, "* g")
        indice = (indice * self._basis).sum(dim=-1).to(torch.int64)

        return indice

    def indice_to_code(self, indice):
        # (..., g)
        indice = (indice.unsqueeze(-1) // self._basis) % self._levels

        code_list = []
        for i in range(self.num_group):
            code = F.embedding(indice[..., i], self.fn((self.codebook[i])))
            code_list.append(code.unsqueeze(-1))
        code = rearrange(torch.cat(code_list, dim=-1), "... d g -> ... (g d)")

        if self.pad:
            code = code[..., : -self.pad]

        return code
