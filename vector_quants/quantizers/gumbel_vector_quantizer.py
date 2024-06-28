import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_embed = cfg.num_embed
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        temp = cfg.temp
        kl_loss_weight = cfg.kl_loss_weight
        # get params end

        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.temp = temp
        self.kl_loss_weight = kl_loss_weight

        # init codebook

        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):
        # get indice
        indice, kl_loss = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute diff
        # diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x) + kl_loss
        codebook_loss = kl_loss
        loss_dict = {
            "codebook_loss": codebook_loss,
        }

        x_quant = x + (x_quant - x).detach()

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent, eps=1e-10):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # n, m
        logits = -compute_dist(latent, self.codebook.weight)
        # n, m
        hard = False if self.training else True
        indice = F.gumbel_softmax(logits, tau=self.temp, dim=-1, hard=hard)
        indice = unpack_one(indice, ps, "* m")

        if self.kl_loss_weight > 0:
            # + kl divergence to the prior (uniform) loss, increase cb usage
            # Note:
            #       KL(P(x), Q(x)) = sum_x (P(x) * log(P(x) / Q(x)))
            #       in this case: P(x) is p, Q(x) is uniform distribution (1 / num_embed)
            p = F.softmax(logits, dim=-1)
            kl_loss = (
                self.kl_loss_weight
                * torch.sum(p * torch.log(p * self.num_embed + eps), dim=-1).mean()
            )
        else:
            kl_loss = torch.tensor(0.0).cuda().float()

        return indice, kl_loss

    def indice_to_code(self, indice):
        return torch.einsum("... m, m d -> ... d", indice, self.codebook.weight)
