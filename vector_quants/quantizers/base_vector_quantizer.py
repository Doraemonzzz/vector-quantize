from abc import ABC, abstractmethod

from torch import nn

from .utils import compute_dist, entropy_loss, pack_one


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

    def entropy_loss(self, latent=None, dist=None):
        assert (
            latent is not None or dist is not None
        ), "At least one of latent or dist needs to be specified."
        if dist is None:
            if isinstance(self.codebook, nn.Module):
                self.codebook.weight
            else:
                self.codebook
            # (b, *, d) -> (n, d)
            latent, ps = pack_one(latent, "* d")
            dist = compute_dist(latent, self.codebook)

        loss = entropy_loss(dist, self.entropy_temperature, self.entropy_loss_type)

        return loss
