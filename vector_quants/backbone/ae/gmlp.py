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


class GMlpLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        embed_dim = cfg.hidden_channels
        num_heads = cfg.num_heads
        seq_len = cfg.seq_len
        norm_type = cfg.norm_type
        channel_act = cfg.channel_act
        causal = cfg.causal
        mid_dim = cfg.mid_dim
        token_mixer = "gmlp"
        channel_mixer = cfg.channel_mixer
        bias = cfg.bias
        # get params end

        self.token_mixer = AUTO_TOKEN_MIXER_MAPPING[token_mixer](
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            bias=bias,
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


class GMlpEncoder(nn.Module):
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
        use_channel_pe = cfg.use_channel_pe
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

        cfg.seq_len = self.patch_embed.num_patch
        self.layers = nn.ModuleList([GMlpLayer(cfg) for i in range(cfg.num_layers)])

        # proj feature
        self.final_norm = AUTO_NORM_MAPPING[norm_type](embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim, bias=bias)

        # use in md lrpe
        self.input_shape = [self.patch_embed.num_h_patch, self.patch_embed.num_w_patch]

        self.use_channel_pe = use_channel_pe
        if self.use_channel_pe:
            self.channel_pe = SinCosPe(
                embed_dim=self.patch_embed.num_patch,
                base=base,
            )

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
        return self.patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        # b c h w -> b n d
        x = self.patch_embed(x)

        if self.use_ape:
            shape = x.shape[1:-1]
            x = self.pe(x, shape)

        if self.use_channel_pe:
            x = rearrange(x, "b n d -> b d n")
            shape = x.shape[1:-1]
            x = self.channel_pe(x, shape)
            x = rearrange(x, "b d n -> b n d")

        for layer in self.layers:
            x = layer(x, self.input_shape)

        x = self.out_proj(self.final_norm(x))

        return x


class GMlpDecoder(nn.Module):
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
        use_init = cfg.use_init
        init_std = cfg.init_std
        use_channel_pe = cfg.use_channel_pe
        # get params end

        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        self.layers = nn.ModuleList([GMlpLayer(cfg) for i in range(cfg.num_layers)])
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

        self.in_proj = nn.Linear(in_dim, embed_dim, bias=bias)

        # use in md lrpe
        self.input_shape = [
            self.reverse_patch_embed.num_h_patch,
            self.reverse_patch_embed.num_w_patch,
        ]

        self.use_channel_pe = use_channel_pe
        if self.use_channel_pe:
            self.channel_pe = SinCosPe(
                embed_dim=self.reverse_patch_embed.num_patch,
                base=base,
            )

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
        return self.reverse_patch_embed.num_patch

    def extra_repr(self):
        return print_module(self)

    def forward(
        self,
        x,
    ):
        x = self.in_proj(x)

        if self.use_ape:
            shape = x.shape[1:-1]
            x = self.pe(x, shape=shape)

        if self.use_channel_pe:
            x = rearrange(x, "b n d -> b d n")
            shape = x.shape[1:-1]
            x = self.channel_pe(x, shape)
            x = rearrange(x, "b d n -> b n d")

        # (b, *)
        for layer in self.layers:
            x = layer(x, self.input_shape)

        x = self.reverse_patch_embed(self.final_norm(x))

        return x