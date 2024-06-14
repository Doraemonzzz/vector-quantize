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
    def codes_to_indices(self, codes):
        pass

    @abstractmethod
    def indices_to_codes(self, indices):
        pass
