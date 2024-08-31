# reference: https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/residual_fsq.py
import torch

from .base_vector_quantizer import BaseVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer


class ResidualFiniteScalarQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_residual = cfg.num_residual
        levels = cfg.levels
        # get params end

        self.num_residual = num_residual
        self.fsq = FiniteScalarQuantizer(cfg)
        self._num_embed = self.fsq.num_embed

        scales = []
        levels = torch.Tensor(levels)
        for i in range(self.num_residual):
            scales.append((levels - 1) ** -i)
        self.register_buffer("scales", torch.stack(scales), persistent=False)

        # init codebook
        self.init_codebook()

    def extra_repr(self):
        return f"(num residual): {self.num_residual}"

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        codebook = self.fsq.indice_to_code(torch.arange(self.num_embed))
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x):
        indice_list = []
        x_quant = torch.zeros_like(x)
        residual = x

        for i in range(self.num_residual):
            residual_quant, indice, loss_dict = self.fsq(residual / self.scales[i])
            residual_quant = residual_quant * self.scales[i]

            # update
            residual = residual - residual_quant.detach()
            x_quant = x_quant + residual_quant
            indice_list.append(indice)

        codebook_loss = torch.tensor(0.0).cuda().float()
        loss_dict = {
            "codebook_loss": codebook_loss,
        }

        indice = torch.cat(indice_list, dim=0)

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent):
        pass

    def indice_to_code(self, indice):
        pass
