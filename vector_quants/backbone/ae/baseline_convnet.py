# credit to https://github.com/duchenzhuang/FSQ-pytorch
from torch import nn


class BaselineConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.in_channels
        hidden_channels = cfg.hidden_channels
        embed_dim = cfg.embed_dim

        blocks = [
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
        ]

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(hidden_channels, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class BaselineConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channels = cfg.embed_dim
        out_channel = cfg.in_channels
        hidden_channels = cfg.hidden_channels

        blocks = [
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, stride=2, padding=1),
        ]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend(
            [
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channel, 1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
