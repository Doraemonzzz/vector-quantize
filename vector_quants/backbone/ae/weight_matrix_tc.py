"""
Use update net after encoder, transformer encoder + resnet decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.utils import AUTO_INIT_MAPPING, AUTO_TOKEN_INIT_MAPPING, print_module

from ..utils import GroupNorm

NUM_GROUPS = 1


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

        self.norm1 = GroupNorm(num_groups=NUM_GROUPS, num_channels=in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=bias
        )

        self.norm2 = GroupNorm(num_groups=NUM_GROUPS, num_channels=out_channels)
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


class ResConvDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # get params start
        channels = cfg.hidden_channels_wt
        num_res_blocks = cfg.num_res_blocks
        channel_multipliers = cfg.channel_multipliers
        embed_dim = cfg.hidden_channels
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

        self.norm = GroupNorm(num_groups=NUM_GROUPS, num_channels=channels)
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


class UpdateNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        num_h_patch = cfg.num_h_patch
        cfg.num_w_patch
        sample_step = cfg.sample_step
        update_net_type = cfg.update_net_type
        in_dim = cfg.hidden_channels
        bias = cfg.bias
        base = cfg.base
        # get params end

        self.sample_step = sample_step
        self.update_net_type = update_net_type
        self.v_dim = num_h_patch
        self.in_dim = in_dim
        self.index = torch.empty(0)

        if self.update_net_type in [
            "additive",
            "cosine",
            "rope",
        ]:
            out_dim = in_dim * 2 * num_h_patch
            if update_net_type == "cosine":
                out_dim = int(in_dim * 1.5 * num_h_patch)
            self.proj = nn.Linear(in_dim, out_dim, bias=bias)
            d = out_dim - in_dim * num_h_patch
            if self.update_net_type == "cosine":
                theta = base ** (
                    -2 / d * torch.arange(d, dtype=torch.int64)
                ).float().reshape(
                    self.in_dim,
                    1,
                    -1,
                )
                self.register_buffer("theta", theta, persistent=False)
            else:
                theta = base ** (
                    -2 / d * torch.arange(d // 2, dtype=torch.int64)
                ).float().reshape(self.in_dim, 1, -1)
                self.register_buffer("theta", theta, persistent=False)

    def forward(self, token):
        if self.update_net_type in ["additive", "cosine", "rope"]:
            kv = self.proj(token)
            kv = rearrange(kv, "b n (h d) -> b h n d", h=self.in_dim)
            b, h, n, d = kv.shape
            k, v = kv.split([d - self.v_dim, self.v_dim], dim=-1)
            k = F.silu(k)
            if self.update_net_type in ["cosine", "rope"]:
                if self.index.shape[0] == 0:
                    self.index = (
                        n
                        - 1
                        - torch.arange(n, device=torch.cuda.current_device()).reshape(
                            -1, 1
                        )
                    )
                theta = self.theta * self.index

                if self.update_net_type == "cosine":
                    cos = torch.cos(theta)
                    sin = torch.sin(theta)
                    k = torch.cat([k * cos, k * sin], dim=-1)
                else:
                    theta = repeat(theta, "... d -> ... (d g)", g=2)
                    cos = torch.cos(theta)
                    sin = torch.sin(theta)
                    k_half = torch.stack(
                        [-k[..., 1::2], k[..., ::2]], dim=-1
                    ).reshape_as(k)
                    k = k * cos + k_half * sin

            weight_matrix = torch.einsum("b h n d, b h n e -> b d e h", k, v)
            weight_matrix = rearrange(weight_matrix, "b h w c -> b c h w")

        return weight_matrix


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        embed_dim = cfg.hidden_channels
        num_heads = cfg.num_heads
        norm_type = cfg.norm_type
        channel_act = cfg.channel_act
        use_lrpe = cfg.use_lrpe
        lrpe_type = cfg.lrpe_type
        base = cfg.theta_base
        causal = cfg.causal
        mid_dim = cfg.mid_dim
        token_mixer = cfg.token_mixer
        channel_mixer = cfg.channel_mixer
        bias = cfg.bias
        norm_type = cfg.norm_type
        # get params end

        self.token_mixer = AUTO_TOKEN_MIXER_MAPPING[token_mixer](
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            use_lrpe=use_lrpe,
            lrpe_type=lrpe_type,
            base=base,
            causal=causal,
            norm_type=norm_type,
        )
        self.channel_mixer = AUTO_CHANNEL_MIXER_MAPPING[channel_mixer](
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            act_fun=channel_act,
            bias=bias,
        )
        self.token_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.channel_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)

    def forward(self, x, shape=None):
        x = x + self.token_mixer(self.token_norm(x), shape=shape)
        x = x + self.channel_mixer(self.channel_norm(x))

        return x


class WMTCEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        image_size = cfg.image_size
        patch_size = cfg.patch_size
        channels = cfg.in_channels
        flatten = False
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        out_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        patch_embed_name = cfg.patch_embed_name
        dct_block_size = cfg.dct_block_size
        use_zigzag = cfg.use_zigzag
        use_freq_patch = cfg.use_freq_patch
        init_std = cfg.init_std
        init_method = cfg.init_method
        token_init_method = cfg.token_init_method
        cfg.patch_merge_size
        cfg.use_channel_pe
        sample_step = cfg.sample_step
        num_extra_token = cfg.num_extra_token
        token_pe_type = cfg.token_pe_type
        token_first = cfg.token_first
        # get params end

        self.patch_embed = AUTO_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            flatten=flatten,
            bias=bias,
            dct_block_size=dct_block_size,
            use_zigzag=use_zigzag,
            use_freq_patch=use_freq_patch,
        )
        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_layers)]
        )

        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim, bias=bias)

        self.num_extra_token = num_extra_token
        if self.num_extra_token > 0:
            assert self.num_extra_token == sample_step
            self.extra_token = nn.Parameter(torch.randn(sample_step, embed_dim))
        else:
            self.extra_token = nn.Parameter(torch.randn(embed_dim))

        self.token_pe_type = token_pe_type
        if self.token_pe_type == "learnable":
            self.token_pe = nn.Parameter(torch.randn(sample_step, embed_dim))

        # use in md lrpe
        self.input_shape = [self.patch_embed.num_h_patch, self.patch_embed.num_w_patch]

        self.sample_step = sample_step
        self.init_std = init_std
        self.token_first = token_first
        self.embed_dim = embed_dim

        cfg.num_h_patch = self.patch_embed.num_h_patch
        cfg.num_w_patch = self.patch_embed.num_w_patch

        self.initialize_weights(init_method, token_init_method)

    def initialize_weights(self, init_method, token_init_method):
        self.apply(AUTO_INIT_MAPPING[init_method])
        AUTO_TOKEN_INIT_MAPPING[token_init_method](
            self.extra_token, std=self.embed_dim**-0.5
        )  # std only use when init_method="titok"

    @property
    def num_patch(self):
        return self.patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b c h w -> b n d
        x = self.patch_embed(x)

        if self.num_extra_token:
            token = repeat(self.extra_token, "n d -> b n d", b=x.shape[0])
        else:
            token = repeat(
                self.extra_token, "d -> b n d", b=x.shape[0], n=self.sample_step
            )

        if self.use_ape:
            if self.token_pe_type == "concat":
                if self.token_first:
                    x = torch.cat((token, x), dim=1)
                else:
                    x = torch.cat((x, token), dim=1)
                shape = x.shape[1:-1]
                x = self.pe(x, shape)
            elif self.token_pe_type == "sincos":
                token = self.pe(token, shape=token.shape[1:-1])
                x = self.pe(x, shape=x.shape[1:-1])
                if self.token_first:
                    x = torch.cat((token, x), dim=1)
                else:
                    x = torch.cat((x, token), dim=1)
            else:
                token = token + self.token_pe
                x = self.pe(x, shape=x.shape[1:-1])
                if self.token_first:
                    x = torch.cat((token, x), dim=1)
                else:
                    x = torch.cat((x, token), dim=1)
        else:
            if self.token_pe_type == "learnable":
                token = token + self.token_pe
                if self.token_first:
                    x = torch.cat((token, x), dim=1)
                else:
                    x = torch.cat((x, token), dim=1)
            else:
                token = self.pe(token, shape=token.shape[1:-1])
                if self.token_first:
                    x = torch.cat((token, x), dim=1)
                else:
                    x = torch.cat((x, token), dim=1)

        for layer in self.layers:
            x = layer(x, self.input_shape)

        if self.token_first:
            # [m n] -> [m]
            x = x[:, : -self.num_patch]
        else:
            # [n m] -> [m]
            x = x[:, self.num_patch :]

        # b n d -> b n e
        x = self.out_proj(self.final_norm(x))

        return x


class WMTCDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        cfg.image_size
        cfg.patch_size
        cfg.in_channels
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        in_dim = cfg.embed_dim
        base = cfg.theta_base
        cfg.norm_type
        cfg.patch_embed_name
        cfg.dct_block_size
        cfg.use_zigzag
        cfg.use_freq_patch
        init_std = cfg.init_std
        init_method = cfg.init_method
        cfg.causal = cfg.decoder_causal
        # get params end

        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_layers)]
        )
        self.in_proj = nn.Linear(in_dim, embed_dim, bias=bias)
        self.update_net = UpdateNet(cfg)
        self.cnn_decoder = ResConvDecoder(cfg)

        self.embed_dim = embed_dim
        self.init_std = init_std

        self.initialize_weights(init_method)

    def initialize_weights(self, init_method):
        self.apply(AUTO_INIT_MAPPING[init_method])

    @property
    def num_patch(self):
        return self.reverse_patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b n d -> b n d
        x = self.in_proj(x)
        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.layers:
            x = layer(x, shape)

        x = self.update_net(x)
        x = self.cnn_decoder(x)

        return x
