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
        # get params end

        self.num_residual = num_residual
        self.fsq = FiniteScalarQuantizer(cfg)
        self._num_embed = self.fsq.num_embed

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
        # residual = x.detach().clone() # v1
        residual = x

        for _ in range(self.num_residual):
            residual_quant, indice, loss_dict = self.fsq(residual)

            # update
            # residual = residual - residual_quant # v1
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
