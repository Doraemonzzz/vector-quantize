import torch.nn as nn

from .utils import GroupNorm


class BasicConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        in_channels = cfg.in_channels
        hidden_channels = cfg.hidden_channels
        embed_dim = cfg.embed_dim
        num_conv_blocks = cfg.num_conv_blocks
        bias = cfg.bias
        # get params start

        # b c h w -> b d h w
        blocks = [
            nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1, bias=bias)
        ]
        for _ in range(num_conv_blocks):
            # b d h w -> b d h/2 w/2
            blocks += [
                GroupNorm(num_groups=32, num_channels=hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
                ),
            ]

        # b d h w -> b e h w
        blocks.append(GroupNorm(num_groups=32, num_channels=hidden_channels))
        blocks.append(nn.SiLU(inplace=True))
        blocks.append(nn.Conv2d(hidden_channels, embed_dim, 1, bias=bias))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class BasicConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        in_channels = cfg.in_channels
        hidden_channels = cfg.hidden_channels
        embed_dim = cfg.embed_dim
        num_conv_blocks = cfg.num_conv_blocks
        bias = cfg.bias
        # get params start

        # b e h w -> b d h w
        blocks = [
            nn.Conv2d(embed_dim, hidden_channels, 3, stride=1, padding=1, bias=bias),
        ]
        for _ in range(num_conv_blocks):
            # b d h w -> b d 2h 2w
            blocks += [
                GroupNorm(num_groups=32, num_channels=hidden_channels),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1, bias=bias
                ),
            ]

        # b d h w -> b e h w
        blocks.append(GroupNorm(num_groups=32, num_channels=hidden_channels))
        blocks.append(nn.SiLU(inplace=True))
        blocks.append(nn.Conv2d(hidden_channels, in_channels, 1, bias=bias))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
