from abc import ABC, abstractmethod

import torch
from torch import nn

from .utils import compute_dist, entropy_loss_fn, kl_loss_fn, pack_one


class BaseVectorQuantizer(ABC, nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @property
    def num_embed(self):
        pass

    @abstractmethod
    def init_codebook(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def latent_to_indice(self, latent):
        pass

    @abstractmethod
    def indice_to_code(self, indice):
        pass

    def get_dist(self, latent):
        if isinstance(self.codebook, nn.Module):
            codebook = self.codebook.weight
        else:
            codebook = self.codebook
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        dist = compute_dist(latent, codebook)

        return dist

    def entropy_loss(self, latent=None, dist=None):
        # E[H(p)] - H[E(p)]
        if not hasattr(self, "entropy_loss_weight") or self.entropy_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        assert (
            latent is not None or dist is not None
        ), "At least one of latent or dist needs to be specified."
        if dist is None:
            dist = self.get_dist(latent)
        loss = entropy_loss_fn(-dist, self.entropy_temperature, self.entropy_loss_type)

        return loss

    def kl_loss(self, latent=None, dist=None):
        # kl(p || uniform) = entropy(p) + c
        if not hasattr(self, "kl_loss_weight") or self.kl_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        assert (
            latent is not None or dist is not None
        ), "At least one of latent or dist needs to be specified."
        if dist is None:
            dist = self.get_dist(latent)

        loss = kl_loss_fn(-dist)

        return loss
