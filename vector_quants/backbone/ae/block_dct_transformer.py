"""
This one does not work well, don't use it.
Use block dct to replace patchify

Notation:
    b: batch_size
    m: number of dct patch
    f: dct_block_size ** 2
    d: embed dim
    e: vq dim
    n: num_patch


First do the following rearrange:
    x = rearrange(x, "b c (h p1) (w p2) -> b (h w c) (p1 p2)", p1=dct_block_size, p2=dct_block_size)
    x: (b m f)

if encoder_transpose_feature:
    x: (b m f) -----------> (b f m) -----------> (b f d) -----------> (b f d) -----------> (b f e)
                rearrange            patch_proj            encoder              out_proj
    if decoder_transpose_feature:
        use shape (b f e) in decoder,
        x: (b f e) -----------> (b e f) -----------> (b f e) -----------> (b f d) -----------> (b f d) -----------> (b f m) -----------> (b m f)
                     rearrange            rearrange            in_proj              decoder            r_patch_proj           rearrange
    else:
        use shape (b e f) in decoder,
        x: (b f e) -----------> (b f d) -----------> (b n d) -----------> (b n d) -----------> (b (h w) c)
                     in_proj             rearrange             decoder            r_patch_proj
                                            to
else: (no need)
    x: (b m f) -----------> (b m d) -----------> (b m d) -----------> (b m e)
                patch_proj            encoder              out_proj
    if decoder_transpose_feature:
        use shape (b e m) in decoder,
        x: (b m e) -----------> (b e m) -----------> (b e d) -----------> (b e d) -----------> (b e m) -----------> (b m e) -----------> (b m f)
                    rearrange             in_proj              decoder                proj              rearrange           r_patch_proj
    else:
        use shape (b m e) in decoder,
        x: (b m e) -----------> (b m d) -----------> (b m d) -----------> (b m f)
                     in_proj              decoder            r_patch_embed
"""


import torch.nn as nn
from einops import rearrange

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
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


class BlockDctTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        image_size = cfg.image_size
        cfg.patch_size
        channels = cfg.in_channels
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        out_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        patch_embed_name = "block_dct"
        dct_block_size = cfg.dct_block_size
        use_zigzag = cfg.use_zigzag
        use_block_dct_only = cfg.use_block_dct_only
        # get params end

        self.patch_embed = AUTO_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            dct_block_size=dct_block_size,
            embed_dim=embed_dim,
            channels=channels,
            bias=bias,
            use_zigzag=use_zigzag,
        )

        self.use_block_dct_only = use_block_dct_only

        if not self.use_block_dct_only:
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

        # dim = embed_dim
        self.out_proj = nn.Linear(embed_dim, out_dim, bias=bias)

        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)

    @property
    def num_patch(self):
        return self.patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b c h w -> b f d
        x = self.patch_embed(x)

        if not self.use_block_dct_only:
            shape = x.shape[1:-1]

            if self.use_ape:
                x = self.pe(x, shape)

            for layer in self.layers:
                x = layer(x, self.input_shape)

        x = self.out_proj(self.final_norm(x))

        return x


class BlockDctTransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        image_size = cfg.image_size
        patch_size = cfg.patch_size
        channels = cfg.in_channels
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        in_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        patch_embed_name = "block_dct"
        dct_block_size = cfg.dct_block_size
        use_zigzag = cfg.use_zigzag
        transpose_feature = cfg.transpose_feature
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

        self.reverse_patch_embed = AUTO_REVERSE_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
            bias=bias,
            dct_block_size=dct_block_size,
            use_zigzag=use_zigzag,
            transpose_feature=transpose_feature,
        )

        self.transpose_feature = transpose_feature

        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)

        # use in md lrpe
        self.input_shape = [
            self.reverse_patch_embed.num_h_patch,
            self.reverse_patch_embed.num_w_patch,
        ]

        if not self.transpose_feature:
            self.dct_block_size = dct_block_size
            self.dct_patch_size = image_size // dct_block_size
            d1 = self.dct_patch_size * self.dct_patch_size * channels
            self.in_proj = nn.Linear(in_dim, d1, bias=bias)
            self.patch_size = patch_size
            d2 = self.patch_size * self.patch_size * channels
            self.proj = nn.Linear(d2, embed_dim, bias=bias)
        else:
            self.in_proj = nn.Linear(in_dim, embed_dim, bias=bias)

    @property
    def num_patch(self):
        return self.reverse_patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):

        # if self.decoder_transpose_feature: b n d -> b d n
        if self.transpose_feature:
            # b f e -> b f d
            x = self.in_proj(x)
        else:
            # b f e -> b f d
            x = self.in_proj(x)
            x = rearrange(
                x,
                "b (p1 p2) (h w c) -> b c (h p1) (w p2)",
                h=self.dct_patch_size,
                w=self.dct_patch_size,
                p1=self.dct_block_size,
                p2=self.dct_block_size,
            )
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            x = self.proj(x)

        shape = x.shape[1:-1]
        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.layers:
            x = layer(x, self.input_shape)

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
