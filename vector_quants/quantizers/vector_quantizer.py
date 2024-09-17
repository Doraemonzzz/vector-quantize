import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import compute_dist, pack_one, unpack_one


class VectorQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_embed = cfg.num_embed
        embed_dim = cfg.embed_dim
        commitment_loss_weight = cfg.commitment_loss_weight
        entropy_temperature = cfg.entropy_temperature
        entropy_loss_type = cfg.entropy_loss_type
        entropy_loss_weight = cfg.entropy_loss_weight
        vq_norm_type = cfg.vq_norm_type
        vq_init_type = cfg.vq_init_type
        # get params end

        self._num_embed = num_embed
        self.embed_dim = embed_dim
        self.commitment_loss_weight = commitment_loss_weight
        self.entropy_temperature = entropy_temperature
        self.entropy_loss_type = entropy_loss_type
        self.entropy_loss_weight = entropy_loss_weight
        self.vq_norm_type = vq_norm_type
        if self.vq_norm_type == "l2":
            self.fn = lambda x: F.normalize(x, p=2, dim=-1)
        elif self.vq_norm_type == "l1":
            self.fn = lambda x: F.normalize(x, p=1, dim=-1)
        else:
            self.fn = lambda x: x
        # init codebook
        self.vq_init_type = vq_init_type
        self.need_init = True
        self.init_codebook(vq_init_type)

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self, vq_init_type):
        self.codebook = nn.Embedding(self.num_embed, self.embed_dim)
        if vq_init_type == "normal":
            nn.init.normal_(self.codebook.weight, mean=0, std=self.embed_dim**-0.5)
        else:
            nn.init.uniform_(
                self.codebook.weight, -1 / self.num_embed, 1 / self.num_embed
            )

    def data_init(self, x):
        # use the first batch to init
        import torch.distributed as dist

        x_flatten = x.reshape(-1, x.shape[-1])
        x_mean = torch.mean(x_flatten, dim=0)
        x_var = torch.mean(x_flatten**2, dim=0)
        num_group = torch.tensor([1.0], device=x.device)
        # all reduce
        dist.all_reduce(x_mean)
        dist.all_reduce(x_var)
        dist.all_reduce(num_group)
        mean = x_mean / num_group
        std = (x_var / num_group) ** 0.5
        # init
        randn = (
            torch.randn(self.num_embed, self.embed_dim, device=x.device) * std + mean
        )
        self.codebook.weight.data.copy_(randn)
        self.need_init = False

    def forward(self, x, use_group_id=False):
        if self.training and self.vq_init_type == "data" and self.need_init:
            self.data_init(x)
        x = self.fn(x)

        # get indice
        indice = self.latent_to_indice(x)

        # quantize
        x_quant = self.indice_to_code(indice)

        # compute codebook loss
        commitment_loss = F.mse_loss(x_quant.detach(), x)
        codebook_loss = (
            F.mse_loss(x_quant, x.detach())
            + self.commitment_loss_weight * commitment_loss
        )

        entropy_loss = self.entropy_loss(x)

        loss_dict = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "entropy_loss": entropy_loss,
        }

        x_quant = x + (x_quant - x).detach()

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent, use_group_id=False):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        # n, m
        dist = compute_dist(latent, self.fn(self.codebook.weight))
        # n, 1
        indice = torch.argmin(dist, dim=-1)
        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice, use_group_id=False):
        if len(indice.shape) >= 3 and indice.shape[-1] == 1:
            indice = indice.squeeze(-1)
        return self.fn(self.codebook(indice))
