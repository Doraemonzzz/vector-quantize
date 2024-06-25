import torch

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import pack_one, unpack_one


class LookUpFreeQuantizer(BaseVectorQuantizer):
    def __init__(self, embed_dim, codebook_value=1):
        super().__init__()
        base = 2
        self.base = base
        num_levels = embed_dim
        levels = [base] * num_levels
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        self.embed_dim = embed_dim
        self.codebook_value = codebook_value

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    @property
    def num_embed(self):
        return self._num_embed

    def forward(self, x):
        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        diff = torch.tensor(0.0).cuda().float()
        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        print(self._basis, self._levels)
        indice = ((latent > 0).int() * self._basis).sum(dim=-1).to(torch.int32)

        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice):
        # (..., d)
        indice = (indice.unsqueeze(-1) // self._basis) % self._levels
        code = torch.where(indice > 0, self.codebook_value, -self.codebook_value)

        return code
