"""
Use block dct at the end of encoder.

Notation:
    b: batch_size
    n: num_patch
    d: embed dim
    e: vq dim
    m: number of dct patch
    f: dct_block_size ** 2

Encoder output: (b n d)
Then do the following transform:
    (b n d) -----------> (b n) = (b (h w)) -----------> (b h w) = (b (h p1) (w p2)) -----------> block dct
    (b (h w) (p1 p2)) = (b m f) -----------> (b f m) -----------> (b f e)

Decoder input: (b f e)
    (b f e) -----------> (b f m) -----------> (b m f) = (b (h w) (p1 p2)) ----------->
    (b (h p1) (w p2)) = (b h w) -----------> (b (h w)) = (b n) -----------> (b n d)

"""

import torch.nn as nn
from einops import rearrange, repeat

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.ops import dct_2d, idct_2d, zigzag_indices
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

    def forward(self, x, shape=None):
        x = x + self.token_mixer(self.token_norm(x), shape=shape)
        x = x + self.channel_mixer(self.channel_norm(x))

        return x


class FeatureDctTransformerEncoder(nn.Module):
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

        # use in md lrpe
        self.input_shape = [self.patch_embed.num_h_patch, self.patch_embed.num_w_patch]

        self.dct_block_size = dct_block_size
        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        num_patch = (
            self.patch_embed.num_h_patch
            * self.patch_embed.num_w_patch
            // (dct_block_size**2)
        )
        self.out_proj = nn.Linear(num_patch, out_dim, bias=bias)

        self.use_zigzag = use_zigzag
        if self.use_zigzag:
            indices, reverse_indices = zigzag_indices(
                self.dct_block_size, self.dct_block_size
            )
            self.register_buffer("indices", indices, persistent=False)

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

        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape)

        for layer in self.layers:
            x = layer(x, self.input_shape)

        # block dct begin
        x = self.final_norm(x)
        x = x.mean(dim=-1)
        x = rearrange(
            x, "b (h w) -> b h w", h=self.input_shape[0], w=self.input_shape[1]
        )
        x = rearrange(
            x,
            "b (h p1) (w p2) -> b h w p1 p2",
            p1=self.dct_block_size,
            p2=self.dct_block_size,
        )
        x = dct_2d(x, norm="ortho")
        x = rearrange(x, "b h w p1 p2 -> b (p1 p2) (h w)")
        if self.use_zigzag:  # take dct coef as seqlen
            x = x[:, self.indices]
        # block dct end

        x = self.out_proj(x)

        return x


class FeatureDctTransformerDecoder(nn.Module):
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
        dct_block_size = cfg.dct_block_size
        use_zigzag = cfg.use_zigzag
        use_freq_patch = cfg.use_freq_patch
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
        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.reverse_patch_embed = AUTO_REVERSE_PATCH_EMBED_MAPPING[patch_embed_name](
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

        # use in md lrpe
        self.input_shape = [
            self.reverse_patch_embed.num_h_patch,
            self.reverse_patch_embed.num_w_patch,
        ]

        self.dct_block_size = dct_block_size
        num_patch = (
            self.reverse_patch_embed.num_h_patch
            * self.reverse_patch_embed.num_w_patch
            // (self.dct_block_size**2)
        )
        self.in_proj = nn.Linear(in_dim, num_patch, bias=bias)

        self.use_zigzag = use_zigzag
        if self.use_zigzag:
            indices, reverse_indices = zigzag_indices(
                self.dct_block_size, self.dct_block_size
            )
            self.register_buffer("reverse_indices", reverse_indices, persistent=False)
        self.embed_dim = embed_dim

    @property
    def num_patch(self):
        return self.reverse_patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b f e -> b f m
        x = self.in_proj(x)

        # block dct begin
        if self.use_zigzag:  # take dct coef as seqlen
            x = x[:, self.inverse_indices]
        h, w = (
            self.input_shape[0] // self.dct_block_size,
            self.input_shape[1] // self.dct_block_size,
        )
        x = rearrange(
            x,
            "b (p1 p2) (h w) -> b h w p1 p2",
            p1=self.dct_block_size,
            p2=self.dct_block_size,
            h=h,
            w=w,
        )
        x = idct_2d(x, norm="ortho")
        x = rearrange(x, "b h w p1 p2 -> b (h p1) (w p2)")
        x = rearrange(x, "b h w -> b (h w)")
        x = repeat(x, "b n -> b n d", d=self.embed_dim)
        # block dct end

        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.layers:
            x = layer(x, self.input_shape)

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
