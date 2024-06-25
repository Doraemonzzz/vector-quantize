import torch

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import round_ste


class RadialQuantizer(BaseVectorQuantizer):
    def __init__(self, base, embed_dim, codebook_value=1):
        super().__init__()
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
        x_quant, indice = self.latent_to_code_and_indice(x)

        diff = torch.tensor(0.0).cuda().float()
        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_code_and_indice(self, latent):
        # x -> sin(x) -> arcsin(sin(x)) -> [-pi/2, pi/2] -> [0, 1] -> [0, c - 1]
        d = self._levels - 1
        number = round_ste(
            (torch.arcsin(torch.sin(latent)) + torch.pi / 2) / torch.pi * d
        )
        # [0, c - 1] -> [0, 1] -> [-1/2, 1/2] -> [-pi/2, pi/2]
        code = torch.sin(torch.pi * (number / d - 0.5))
        indice = (number * self._basis).sum(dim=-1).to(torch.int32)

        return code, indice

    def latent_to_indice(self, latent):
        pass

    def indice_to_code(self, indice):
        pass
