from abc import ABC, abstractmethod

from torch import nn


class BaseVectorQuantizer(ABC, nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    @property
    def num_embed(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def latent_to_indice(self, latent):
        pass

    @abstractmethod
    def indice_to_code(self, indices):
        pass
