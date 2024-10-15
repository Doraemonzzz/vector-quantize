import torch
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import round_ste


class RadialQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        base = cfg.base
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        # get params end

        self.base = base
        num_levels = embed_dim
        levels = [base] * num_levels
        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("_basis", _basis, persistent=False)

        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight

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
        # x -> sin(x) -> arcsin(sin(x)) -> [-pi/2, pi/2] -> [0, 1] -> [0, c - 1]
        d = self._levels - 1
        number = round_ste(F.sigmoid(latent) * d)
        # [0, c - 1] -> [0, 1] -> [-1/2, 1/2] -> [-pi/2, pi/2]
        code = torch.sin(torch.pi * (number / d - 0.5))
        # code = number / d
        indice = (number * self._basis).sum(dim=-1).to(torch.int64)

        return code, indice

    def latent_to_indice(self, latent):
        pass

    def indice_to_code(self, indice):
        pass
