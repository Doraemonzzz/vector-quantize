import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class EMAVectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        num_embed,
        embed_dim,
        commitment_loss_weight=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()
        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.decay = decay
        self.epsilon = epsilon

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        # ema parameters
        # ema usage count: total count of each embedding trough epochs
        self.register_buffer("ema_count", torch.zeros(num_embed))

        # same size as dict, initialized as codebook
        # the updated means
        self.register_buffer("ema_weight", torch.empty((num_embed, embed_dim)))
        self.ema_weight.data.uniform_(-1 / num_embed, 1 / num_embed)

        self.init_codebook()

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        nn.init.uniform_(self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed)
        self.codebook.requires_grad_(False)

    def forward(self, x):
        # get indices
        indices, embed_count_sum, embed_sum = self.codes_to_indices(x)

        # quantize
        x_quant = self.indices_to_codes(indices)

        if self.training:
            ema_count = self.decay * self.ema_count + (1 - self.decay) * embed_count_sum
            # Laplace smoothing of the ema count(avoid zero)
            N = embed_count_sum.sum().item()
            self.ema_count = (
                (ema_count + self.epsilon) / (N + self.num_embed * self.epsilon) * N
            )

            self.ema_weight = (
                self.decay * self.ema_weight + (1 - self.decay) * embed_sum
            )
            self.codebook.weight.data = self.ema_weight / self.ema_count.unsqueeze(-1)

        # compute diff
        diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indices

    def codes_to_indices(self, codes):
        # (b, *, d) -> (n, d)
        codes, ps = pack_one(codes, "* d")
        # n, m
        dist = compute_dist(codes, self.codebook.weight)
        # n, 1
        indices = torch.argmin(dist, dim=-1)
        if self.training:
            # n, 1 -> n, V
            indices_onehot = F.one_hot(indices, self.num_embed)
            # n, V -> V
            embed_count_sum = indices_onehot.sum(0)
            # (V, n), (n, d) -> (V, n)
            embed_sum = indices_onehot.transpose(0, 1).to(torch.float32) @ codes
            # torch.distributed.all_reduce(
            #     embed_count_sum,
            # )
            # torch.distributed.all_reduce(
            #     embed_sum,
            # )
        else:
            embed_count_sum = None
            embed_sum = None

        indices = unpack_one(indices, ps, "*")

        return indices, embed_count_sum, embed_sum

    def indices_to_codes(self, indices):
        return self.codebook(indices)