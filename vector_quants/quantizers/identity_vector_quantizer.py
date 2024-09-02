# no vector quantize, only for align the interface

import torch

from .base_vector_quantizer import BaseVectorQuantizer


class IdentityVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self._num_embed = 1

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        return

    def forward(self, x, use_group_id=False):
        # quantize
        x_quant = x

        # get indice
        indice = torch.tensor([0]).cuda()

        loss_dict = {
            "codebook_loss": torch.tensor(0.0).cuda().float(),
        }

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent, use_group_id=False):
        return

    def indice_to_code(self, indice, use_group_id=False):
        return
