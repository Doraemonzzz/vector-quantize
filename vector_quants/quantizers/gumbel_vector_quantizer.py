import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embed,
        embed_dim,
        commitment_loss_weight=0.25,
        temp=1,
        kl_loss_weight=5e-4,
    ):
        super().__init__()
        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.temp = temp
        self.kl_loss_weight = kl_loss_weight

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)

    def forward(self, x):
        # get indices
        indices, kl_loss = self.codes_to_indices(x)

        # quantize
        x_quant = self.indices_to_codes(indices)

        # compute diff
        diff = (
            F.mse_loss(x_quant, x.detach())
            + self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)
            + kl_loss
        )

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indices

    def codes_to_indices(self, codes, eps=1e-10):
        # (b, *, d) -> (n, d)
        codes, ps = pack_one(codes, "* d")
        # n, m
        dist = compute_dist(codes, self.codebook.weight)
        # n, m
        hard = False if self.training else True
        indices = F.gumbel_softmax(dist, tau=self.temp, dim=-1, hard=hard)
        indices = unpack_one(indices, ps, "* m")

        if self.kl_loss_weight > 0:
            # + kl divergence to the prior (uniform) loss, increase cb usage
            # Note:
            #       KL(P(x), Q(x)) = sum_x (P(x) * log(P(x) / Q(x)))
            #       in this case: P(x) is p, Q(x) is uniform distribution (1 / num_embed)
            p = F.softmax(dist, dim=-1)
            kl_loss = (
                self.kl_loss_weight
                * torch.sum(p * torch.log(p * self.num_embed + eps), dim=-1).mean()
            )
        else:
            kl_loss = torch.tensor(0.0).cuda().float()

        return indices, kl_loss

    def indices_to_codes(self, indices):
        return torch.einsum("... m, m d -> ... d", indices, self.codebook.weight)
