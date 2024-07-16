import torch
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import pack_one, round_ste, unpack_one


class SequentialScalarQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        cfg.base
        cfg.num_levels
        # get params end

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor(levels), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        self.embed_dim = self._levels.shape[0]

        # init codebook
        self.init_codebook()

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        codebook = self.indice_to_code(torch.arange(self.num_embed))
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x):
        x_quant, indice = self.latent_to_code_and_indice(x)
        # compute codebook loss
        codebook_loss = torch.tensor(0.0).cuda().float()
        loss_dict = {
            "codebook_loss": codebook_loss,
        }

        return x_quant, indice, loss_dict

    def latent_to_code_and_indice(self, latent):
        # wip
        number = 0
        base = 0
        code_list = []
        for i in range(self.num_levels):
            d = self._levels[i] - 1
            number = base * number + round_ste(F.sigmoid(latent[:, i : i + 1]) * d)
            code_list.append(number)
            code = number / self._basis[i]
            base = d

        code = torch.cat(code_list, dim=-1)
        indice = number

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
