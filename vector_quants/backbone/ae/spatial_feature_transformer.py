"""
Do feature attention at the end of at each layer
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


class SFTransformerEncoder(nn.Module):
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
        use_init = cfg.use_init
        init_std = cfg.init_std
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
        self.spatial_layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_layers)]
        )

        # proj feature
        self.norm1 = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.proj1 = nn.Linear(embed_dim, out_dim, bias=bias)
        n = self.patch_embed.num_h_patch * self.patch_embed.num_w_patch
        # proj sequence
        self.norm2 = AUTO_NORM_MAPPING[norm_type](n)
        self.proj2 = nn.Linear(n, embed_dim, bias=bias)
        self.feature_layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_feature_layers)]
        )

        # use in md lrpe
        self.input_shape = [self.patch_embed.num_h_patch, self.patch_embed.num_w_patch]

        self.use_init = use_init
        self.init_std = init_std
        self.embed_dim = embed_dim

        if self.use_init:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @property
    def num_patch(self):
        return self.embed_dim

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b c h w -> b n d
        x = self.patch_embed(x)

        shape = x.shape[1:-1]

        # spatial attn mixing
        if self.use_ape:
            x = self.pe(x, shape)

        for layer in self.spatial_layers:
            x = layer(x, self.input_shape)

        x = self.proj1(self.norm1(x))

        # feature attn mixing
        x = rearrange(x, "b n d -> b d n")
        x = self.proj2(self.norm2(x))

        if self.use_ape:
            x = self.pe(x, shape)

        for layer in self.feature_layers:
            x = layer(x)

        x = rearrange(x, "b d n -> b n d")

        return x


class SFTransformerDecoder(nn.Module):
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
        out_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        patch_embed_name = cfg.patch_embed_name
        dct_block_size = cfg.dct_block_size
        use_zigzag = cfg.use_zigzag
        use_freq_patch = cfg.use_freq_patch
        use_init = cfg.use_init
        init_std = cfg.init_std
        # get params end

        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        self.spatial_layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_layers)]
        )

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

        n = self.reverse_patch_embed.num_h_patch * self.reverse_patch_embed.num_w_patch
        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.norm1 = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.proj1 = nn.Linear(embed_dim, n, bias=bias)
        self.feature_layers = nn.ModuleList(
            [TransformerLayer(cfg) for i in range(cfg.num_feature_layers)]
        )
        self.norm2 = AUTO_NORM_MAPPING[norm_type](out_dim)
        self.proj2 = nn.Linear(out_dim, embed_dim, bias=bias)

        # use in md lrpe
        self.input_shape = [
            self.reverse_patch_embed.num_h_patch,
            self.reverse_patch_embed.num_w_patch,
        ]

        self.embed_dim = embed_dim
        self.use_init = use_init
        self.init_std = init_std

        if self.use_init:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    @property
    def num_patch(self):
        return self.embed_dim

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        x = rearrange(x, "b n d -> b d n")
        # feature attn mixing
        shape = x.shape[1:-1]
        if self.use_ape:
            x = self.pe(x, shape)

        for layer in self.feature_layers:
            x = layer(x)

        x = self.proj1(self.norm1(x))

        x = rearrange(x, "b d n -> b n d")

        # spatial attn mixing
        x = self.proj2(self.norm2(x))

        shape = x.shape[1:-1]
        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.spatial_layers:
            x = layer(x, self.input_shape)

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
