"""
Use update net after encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.utils import print_module


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
                    self.index = n - torch.arange(
                        n, device=torch.cuda.current_device()
                    ).reshape(-1, 1)
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
            weight_matrix = rearrange(weight_matrix, "b n m d -> b (n m) d")

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


class WeightMatrixTransformerEncoderV2(nn.Module):
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
        cfg.patch_merge_size
        cfg.use_channel_pe
        sample_step = cfg.sample_step
        num_extra_token = cfg.num_extra_token
        token_pe_type = cfg.token_pe_type
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
        self.use_init = use_init
        self.init_std = init_std

        if self.use_init:
            self.initialize_weights()

        cfg.num_h_patch = self.patch_embed.num_h_patch
        cfg.num_w_patch = self.patch_embed.num_w_patch

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

        if self.num_extra_token:
            token = repeat(self.extra_token, "n d -> b n d", b=x.shape[0])
        else:
            token = repeat(
                self.extra_token, "d -> b n d", b=x.shape[0], n=self.sample_step
            )

        if self.use_ape:
            if self.token_pe_type == "concat":
                x = torch.cat((x, token), dim=1)
                shape = x.shape[1:-1]
                x = self.pe(x, shape)
            elif self.token_pe_type == "sincos":
                token = self.pe(token, shape=token.shape[1:-1])
                x = self.pe(x, shape=x.shape[1:-1])
                x = torch.cat((x, token), dim=1)
            else:
                token = token + self.token_pe
                x = self.pe(x, shape=x.shape[1:-1])
                x = torch.cat((x, token), dim=1)
        else:
            if self.token_pe_type == "learnable":
                token = token + self.token_pe
                x = torch.cat((x, token), dim=1)
            else:
                token = self.pe(token, shape=token.shape[1:-1])
                x = torch.cat((x, token), dim=1)

        for layer in self.layers:
            x = layer(x, self.input_shape)

        x = x[:, self.num_patch :]

        # b n d -> b n e
        x = self.out_proj(self.final_norm(x))

        return x


class WeightMatrixTransformerDecoderV2(nn.Module):
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

        self.update_net = UpdateNet(cfg)

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
        # b n d -> b n d
        x = self.in_proj(x)
        shape = x.shape[1:-1]

        if self.use_ape:
            x = self.pe(x, shape=shape)

        # (b, *)
        for layer in self.layers:
            x = layer(x, shape)

        x = self.update_net(x)

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
