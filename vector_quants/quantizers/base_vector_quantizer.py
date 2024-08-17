from abc import ABC, abstractmethod

import torch
from torch import nn

from .utils import (
    compute_dist,
    entropy_loss_fn,
    kl_loss_fn,
    pack_one,
    sample_entropy_loss_fn,
)


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
    def latent_to_indice(self, latent, use_group_id=True):
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

    def entropy_loss(self, latent=None, logits=None, is_distance=True):
        # E[H(p)] - H[E(p)]
        if not hasattr(self, "entropy_loss_weight") or self.entropy_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        else:
            assert (
                latent is not None or logits is not None
            ), "At least one of latent or dist needs to be specified."
            if logits is None:
                logits = self.get_dist(latent)
            # if we get logits by distance, we use -logits as logits, else, we use logits
            flag = -1 if is_distance else 1
            loss = entropy_loss_fn(
                flag * logits, self.entropy_temperature, self.entropy_loss_type
            )

        return loss

    def sample_entropy_loss(self, latent=None, logits=None, is_distance=True):
        # E[H(p)]
        if (
            not hasattr(self, "sample_entropy_loss_weight")
            or self.sample_entropy_loss_weight == 0
        ):
            loss = torch.tensor(0.0).cuda().float()
        else:
            assert (
                latent is not None or logits is not None
            ), "At least one of latent or dist needs to be specified."
            if logits is None:
                logits = self.get_dist(latent)
            # if we get logits by distance, we use -logits as logits, else, we use logits
            flag = -1 if is_distance else 1
            loss = sample_entropy_loss_fn(
                flag * logits, self.entropy_temperature, self.entropy_loss_type
            )

        return loss

    def codebook_entropy_loss(self, latent=None, logits=None, is_distance=True):
        # - H[E(p)]
        if (
            not hasattr(self, "codebook_entropy_loss_weight")
            or self.codebook_entropy_loss_weight == 0
        ):
            loss = torch.tensor(0.0).cuda().float()
        else:
            assert (
                latent is not None or logits is not None
            ), "At least one of latent or dist needs to be specified."
            if logits is None:
                logits = self.get_dist(latent)
            # if we get logits by distance, we use -logits as logits, else, we use logits
            flag = -1 if is_distance else 1
            loss = sample_entropy_loss_fn(
                flag * logits, self.entropy_temperature, self.entropy_loss_type
            )

        return loss

    def kl_loss(self, logits):
        # kl(p || uniform) = -entropy(p) + c
        if not hasattr(self, "kl_loss_weight") or self.kl_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        else:
            loss = kl_loss_fn(logits)

        return loss
