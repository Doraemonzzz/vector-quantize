import torch
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import pack_one, round_ste, unpack_one


class FiniteScalarQuantizer(BaseVectorQuantizer):
    def __init__(self, levels):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        self.embed_dim = self._levels.shape[0]

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    @property
    def num_embed(self):
        return self._num_embed

    def forward(self, x):
        x_quant, indice = self.latent_to_code_and_indice(x)
        diff = torch.tensor(0.0).cuda().float()

        return x_quant, diff, indice

    def latent_to_code_and_indice(self, latent):
        d = self._levels - 1
        number = round_ste(F.sigmoid(latent) * d)
        code = number / d
        indice = (number * self._basis).sum(dim=-1).to(torch.int32)

        return code, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        number = round_ste(F.sigmoid(latent) * (self._levels - 1))
        indice = (number * self._basis).sum(dim=-1).to(torch.int32)

        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice):
        # (..., d)
        code = (indice.unsqueeze(-1) // self._basis) % self._levels
        # convert to [0, 1]
        code = code / (self._levels - 1)

        return code

    # ###### v1
    # def forward(self, x):
    #     # get indice
    #     indice = self.latent_to_indice(x)

    #     # quantize
    #     x_quant = self.indice_to_code(indice)

    #     diff = torch.tensor(0.0).cuda().float()

    #     return x_quant, diff, indice

    # def latent_to_indice(self, latent):
    #     # (b, *, d) -> (n, d)
    #     latent, ps = pack_one(latent, "* d")
    #     number = round_ste(F.sigmoid(latent) * (self._levels - 1))
    #     indice = (number * self._basis).sum(dim=-1).to(torch.int32)

    #     indice = unpack_one(indice, ps, "*")

    #     return indice

    # def indice_to_code(self, indice):
    #     # (..., d)
    #     code = (indice.unsqueeze(-1) // self._basis) % self._levels
    #     # convert to [0, 1]
    #     code = code / (self._levels - 1)

    #     return code
    # ###### v1

    # ###### V2
    # def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
    #     """Bound `z`, an array of shape (..., d)."""
    #     d = self._levels - 1
    #     # [0, L - 1]
    #     output = d * F.sigmoid(z)
    #     return output

    # def quantize(self, z: Tensor) -> Tensor:
    #     """Quantizes z, returns quantized zhat, same shape as z."""
    #     d = self._levels - 1
    #     # [0, L - 1] -> [0, 1]
    #     quantized = round_ste(self.bound(z)) / d
    #     return quantized

    # def _scale(self, zhat_normalized: Tensor) -> Tensor:
    #     d = self._levels - 1
    #     return zhat_normalized * d

    # def _scale_inverse(self, zhat: Tensor) -> Tensor:
    #     d = self._levels - 1
    #     return zhat / d

    # def codes_to_indices(self, zhat: Tensor) -> Tensor:
    #     """Converts a `code` to an index in the codebook."""
    #     zhat = self._scale(zhat)
    #     return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    # def indices_to_codes(self, indices: Tensor) -> Tensor:
    #     """Inverse of `codes_to_indices`."""

    #     indices = rearrange(indices, "... -> ... 1")
    #     codes_non_centered = (indices // self._basis) % self._levels
    #     codes = self._scale_inverse(codes_non_centered)

    #     return codes

    # def forward(self, z: Tensor) -> Tensor:
    #     codes = self.quantize(z)
    #     indices = self.codes_to_indices(codes)
    #     diff = torch.tensor(0.0).cuda().float()

    #     return codes, diff, indices
