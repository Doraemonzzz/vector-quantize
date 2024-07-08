import torch.nn as nn

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.ops import zigzag_indices
from vector_quants.utils import print_module


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
        # get params end

        self.token_mixer = AUTO_TOKEN_MIXER_MAPPING[token_mixer](
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            use_lrpe=use_lrpe,
            lrpe_type=lrpe_type,
            base=base,
            causal=causal,
        )
        self.channel_mixer = AUTO_CHANNEL_MIXER_MAPPING[channel_mixer](
            embed_dim=embed_dim,
            mid_dim=mid_dim,
            act_fun=channel_act,
            bias=bias,
        )
        self.token_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.channel_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)

    def forward(self, x):
        x = x + self.token_mixer(self.token_norm(x))
        x = x + self.channel_mixer(self.channel_norm(x))

        return x


class FreqTransformerEncoder(nn.Module):
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
        # get params end

        self.patch_embed = AUTO_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            flatten=flatten,
            bias=bias,
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

        indices, reverse_indices = zigzag_indices(
            self.patch_embed.num_h_patch, self.patch_embed.num_w_patch
        )
        self.register_buffer("indices", indices, persistent=False)

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # # v1
        # # (b c h w)
        # x = self.patch_embed(x)
        # x = rearrange(dct_2d(x), "b c h w -> b (h w) c")
        # # convert to zigzag order
        # x = x[:, self.indices, :]

        # # v2
        # # (b c h w)
        # x = self.patch_embed(dct_2d(x))
        # x = rearrange(x, "b c h w -> b (h w) c")

        # # convert to zigzag order
        # x = x[:, self.indices, :]

        x = self.patch_embed(x)

        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape)

        for layer in self.layers:
            x = layer(
                x,
            )

        x = self.out_proj(self.final_norm(x))

        return x


class FreqTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        image_size = cfg.image_size
        patch_size = cfg.patch_size
        channels = cfg.in_channels
        flatten = True
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        in_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        patch_embed_name = cfg.patch_embed_name
        # get params end

        self.in_proj = nn.Linear(in_dim, embed_dim, bias=bias)
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
        self.reverse_patch_embed = AUTO_REVERSE_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            flatten=flatten,
            bias=bias,
        )
        indices, reverse_indices = zigzag_indices(
            self.reverse_patch_embed.num_h_patch, self.reverse_patch_embed.num_w_patch
        )
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        x = self.in_proj(x)
        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.layers:
            x = layer(
                x,
            )

        # # v1
        # x = idct_2d(x[:, self.reverse_indices])

        # x = self.reverse_patch_embed(self.final_norm(x))

        # # v2
        # x = x[:, self.reverse_indices]

        # x = self.reverse_patch_embed(self.final_norm(x))

        # x = idct_2d(x)

        # v3
        x = self.reverse_patch_embed(self.final_norm(x))

        return x
