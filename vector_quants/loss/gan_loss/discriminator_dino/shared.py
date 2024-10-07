# credit to: https://github.com/autonomousvision/stylegan-t
"""Shared architecture blocks."""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class MLP(nn.Module):
    def __init__(
        self,
        features_list: list[int],  # Number of features in each layer of the MLP.
        activation: str = "linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float = 1.0,  # Learning rate multiplier.
        linear_out: bool = False,  # Use the 'linear' activation function for the output layer?
    ):
        super().__init__()
        num_layers = len(features_list) - 1
        self.num_layers = num_layers
        self.out_dim = features_list[-1]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            if linear_out and idx == num_layers - 1:
                activation = "linear"
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """if x is sequence of tokens, shift tokens to batch and apply MLP to all"""
        shift2batch = x.ndim == 3

        if shift2batch:
            B, K, C = x.shape
            x = x.flatten(0, 1)

        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        if shift2batch:
            x = x.reshape(B, K, -1)

        return x
