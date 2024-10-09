import torch
import torch.nn as nn
from einops import repeat

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


class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        image_size = cfg.image_size
        patch_size = cfg.patch_size
        channels = cfg.in_channels
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        out_dim = cfg.embed_dim
        base = cfg.theta_base
        num_extra_token = cfg.num_extra_token
        norm_type = cfg.norm_type
        patch_embed_name = cfg.patch_embed_name
        use_init = cfg.use_init
        init_std = cfg.init_std
        mask_ratio = cfg.mask_ratio
        # get params end
        assert not (
            num_extra_token > 0 and mask_ratio > 0
        ), f"Only support: num_extra_token and mask_ratio cant be both > 0, but got: num_extra_token={num_extra_token}, mask_ratio={mask_ratio}"

        self.patch_embed = AUTO_PATCH_EMBED_MAPPING[patch_embed_name](
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            channels=channels,
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

        self.num_extra_token = num_extra_token
        if self.num_extra_token:
            self.extra_token = nn.Parameter(
                torch.randn(self.num_extra_token, embed_dim)
            )
            self.extra_token_pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        # use in md lrpe
        self.input_shape = [self.patch_embed.num_h_patch, self.patch_embed.num_w_patch]

        self.use_init = use_init
        self.init_std = init_std
        self.mask_ratio = mask_ratio

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

    def random_masking(self, x, mask_ratio):
        # credit to: https://github.com/facebookresearch/mae
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [b, n, d], sequence
        """
        b, n, d = x.shape  # batch, length, dim
        len_keep = int(n * (1 - mask_ratio))

        noise = torch.rand(b, n, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(
        self,
        x,
    ):
        # b c h w -> b n d
        x = self.patch_embed(x)

        if self.use_ape:
            x = self.pe(x, self.input_shape)

        if self.num_extra_token > 0:
            extra_token = repeat(self.extra_token, "n d -> b n d", b=x.shape[0])
            extra_token_shape = (self.num_extra_token,)
            extra_token = self.extra_token_pe(extra_token, extra_token_shape)
            x = torch.cat([extra_token, x], dim=1)

        if self.training and self.mask_ratio:
            x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        for layer in self.layers:
            x = layer(x, self.input_shape)

        if self.num_extra_token > 0:
            x = x[:, : self.num_extra_token]

        x = self.out_proj(self.final_norm(x))

        return x


class TransformerDecoder(nn.Module):
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
        num_extra_token = cfg.num_extra_token
        norm_type = cfg.norm_type
        patch_embed_name = cfg.patch_embed_name
        use_init = cfg.use_init
        init_std = cfg.init_std
        mask_ratio = cfg.mask_ratio
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
            bias=bias,
        )

        self.num_extra_token = num_extra_token
        self.mask_ratio = mask_ratio
        if self.num_extra_token > 0 or self.mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.randn(embed_dim))
            if self.num_extra_token > 0:
                self.extra_token_pe = SinCosPe(
                    embed_dim=embed_dim,
                    base=base,
                )

        # use in md lrpe
        self.input_shape = [
            self.reverse_patch_embed.num_h_patch,
            self.reverse_patch_embed.num_w_patch,
        ]
        self.num_token = self.input_shape[0] * self.input_shape[1]

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
        # b n d
        x = self.in_proj(x)
        b = x.shape[0]
        n = x.shape[1]

        if self.num_extra_token > 0 or self.mask_ratio > 0:
            n = self.num_token
            # if num extra token > 0, we use mask token to reconstruct
            # see the difference between repeat and expand: https://github.com/arogozhnikov/einops/issues/202
            mask_tokens = repeat(self.mask_token, "d -> b n d", b=b, n=n)

            if self.use_ape:
                mask_tokens = self.pe(mask_tokens, shape=mask_tokens.shape[1:-1])

            if self.num_extra_token > 0:
                extra_token_shape = (self.num_extra_token,)
                x = self.extra_token_pe(x, extra_token_shape)
            x = torch.cat([x, mask_tokens], dim=1)
        else:
            if self.use_ape:
                x = self.pe(x, shape=self.input_shape)

        # (b, *)
        for layer in self.layers:
            x = layer(x, self.input_shape)

        if self.num_extra_token > 0:
            x = x[:, self.num_extra_token :]

        if self.mask_ratio > 0:
            x = x[:, -self.num_token :]

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
