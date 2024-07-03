# credit to https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/master/vqvae/modules/autoencoder.py
# similar to the arch in magvit

import torch.nn.functional as F
from torch import nn

from .utils import GroupNorm


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, bias: bool = False):
        """
        :param in_channels: input channels of the residual block
        :param out_channels: if None, use in_channels. Else, adds a 1x1 conv layer.
        """
        super().__init__()

        if out_channels is None or out_channels == in_channels:
            out_channels = in_channels
            self.conv_shortcut = None
        else:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding="same", bias=bias
            )

        self.norm1 = GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=bias
        )

        self.norm2 = GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same", bias=bias
        )

    def forward(self, x):

        residual = F.silu(self.norm1(x))
        residual = self.conv1(residual)

        residual = F.silu(self.norm2(residual))
        residual = self.conv2(residual)

        if self.conv_shortcut is not None:
            # contiguous prevents warning:
            # https://github.com/pytorch/pytorch/issues/47163
            # https://discuss.pytorch.org/t/why-does-pytorch-prompt-w-accumulate-grad-h-170-warning-grad-and-param-do-not-obey-the-gradient-layout-contract-this-is-not-an-error-but-may-impair-performance/107760
            x = self.conv_shortcut(x.contiguous())

        return x + residual


class Downsample(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        res = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return res


class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        scale_factor: float = 2.0,
        mode: str = "nearest-exact",
        bias: bool = False,
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding="same", bias=bias
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class ResConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        channels = cfg.hidden_channels
        num_res_blocks = cfg.num_res_blocks
        channel_multipliers = cfg.channel_multipliers
        embed_dim = cfg.embed_dim
        bias = cfg.bias
        # get params end

        self.conv_in = nn.Conv2d(
            3, channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        blocks = []
        ch_in = channels

        for i in range(len(channel_multipliers)):

            ch_out = channels * channel_multipliers[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, bias))
                ch_in = ch_out

            blocks.append(Downsample())

        self.blocks = nn.Sequential(*blocks)

        self.final_residual = nn.Sequential(
            *[ResBlock(ch_in, ch_in, bias) for _ in range(num_res_blocks)]
        )

        self.norm = GroupNorm(num_groups=32, num_channels=ch_in)
        self.conv_out = nn.Conv2d(
            ch_in, embed_dim, kernel_size=1, padding="same", bias=bias
        )

    def forward(self, x):

        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.final_residual(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class ResConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # get params start
        channels = cfg.hidden_channels
        num_res_blocks = cfg.num_res_blocks
        channel_multipliers = cfg.channel_multipliers
        embed_dim = cfg.embed_dim
        bias = cfg.bias
        # get params end

        ch_in = channels * channel_multipliers[-1]

        self.conv_in = nn.Conv2d(
            embed_dim, ch_in, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.initial_residual = nn.Sequential(
            *[ResBlock(ch_in, ch_in, bias) for _ in range(num_res_blocks)]
        )

        blocks = []
        for i in reversed(range(len(channel_multipliers))):
            blocks.append(Upsample(ch_in))
            ch_out = channels * channel_multipliers[i - 1] if i > 0 else channels

            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch_in, ch_out, bias))
                ch_in = ch_out

        self.blocks = nn.Sequential(*blocks)

        self.norm = GroupNorm(num_groups=32, num_channels=channels)
        self.conv_out = nn.Conv2d(
            channels, 3, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.initial_residual(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x
