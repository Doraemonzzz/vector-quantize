# encoder: convnet + attention + extra_token
# decoder: transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.utils import (
    AUTO_INIT_MAPPING,
    AUTO_TOKEN_INIT_MAPPING,
    get_activation_fn,
    print_module,
)

from ..utils import GroupNorm

NUM_GROUPS = 1


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


class UpdateNetV2(nn.Module):
    # low rank decompose
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
        update_net_act = cfg.update_net_act
        update_net_use_conv = cfg.update_net_use_conv
        # get params end

        self.sample_step = sample_step
        self.update_net_type = update_net_type
        self.v_dim = num_h_patch
        self.in_dim = in_dim
        self.act = get_activation_fn(update_net_act)
        self.index = torch.empty(0)
        self.update_net_use_conv = update_net_use_conv

        if self.update_net_use_conv:
            self.conv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=bias)

        if self.update_net_type in [
            "additive",
            "cosine",
            "rope",
            "di_decay",
            "dd_decay",
            "dd_share_decay",
            "additive_decay",
            "delta",
        ]:
            out_dim = 2 * num_h_patch
            if update_net_type == "cosine":
                out_dim = int(1.5 * num_h_patch)
            self.kv_proj = nn.Linear(in_dim, out_dim, bias=bias)
            self.kv_h_proj = nn.Linear(in_dim, 2 * in_dim, bias=bias)
            d = (out_dim - num_h_patch) * in_dim
            if self.update_net_type == "cosine":
                theta = base ** (
                    -2 / d * torch.arange(d, dtype=torch.int64)
                ).float().reshape(
                    self.in_dim,
                    1,
                    -1,
                )
                self.register_buffer("theta", theta, persistent=False)
            elif self.update_net_type == "rope":
                theta = base ** (
                    -2 / d * torch.arange(d // 2, dtype=torch.int64)
                ).float().reshape(self.in_dim, 1, -1)
                self.register_buffer("theta", theta, persistent=False)
            elif self.update_net_type == "di_decay":
                # h, 1, 1
                log_decay = (
                    -torch.arange(1, in_dim + 1, device=torch.cuda.current_device())
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    * 8.0
                    / in_dim
                )
                self.register_buffer("log_decay", log_decay, persistent=False)
            elif self.update_net_type == "dd_decay":
                self.decay_proj = nn.Linear(in_dim, num_h_patch, bias=bias)
                self.decay_h_proj = nn.Linear(in_dim, in_dim, bias=bias)
            elif self.update_net_type == "dd_share_decay":
                self.decay_h_proj = nn.Linear(in_dim, in_dim, bias=bias)
            elif self.update_net_type == "delta":
                self.beta_proj = nn.Linear(in_dim, in_dim, bias=bias)

    def forward(self, token):
        kv = self.kv_proj(token)
        # b n h
        kv_h = self.kv_h_proj(token)
        b, n, d = kv.shape
        k, v = kv.split([d - self.v_dim, self.v_dim], dim=-1)
        k_h, v_h = kv_h.chunk(2, dim=-1)
        k = torch.einsum("b n d, b n h -> b h n d", k, k_h)
        v = torch.einsum("b n d, b n h -> b h n d", v, v_h)

        if self.update_net_type in [
            "additive",
            "cosine",
            "rope",
            "di_decay",
            "dd_decay",
            "dd_share_decay",
            "additive_decay",
        ]:
            if self.update_net_type not in ["additive_decay"]:
                k = self.act(k)
            else:
                k = F.softmax(k, dim=-2)

            v = self.act(v)
            if self.update_net_type in ["cosine", "rope"]:
                if self.index.shape[0] == 0:
                    self.index = (
                        n
                        - 1
                        - torch.arange(
                            n, dtype=torch.int64, device=torch.cuda.current_device()
                        ).reshape(-1, 1)
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
            elif self.update_net_type == "di_decay":
                # kv = lambda_ * kv + ki * vi
                if self.index.shape[0] == 0:
                    self.index = (
                        n
                        - 1
                        - torch.arange(
                            n, dtype=torch.int64, device=torch.cuda.current_device()
                        ).reshape(-1, 1)
                    )
                log_decay = self.index.float() * self.log_decay.float()
                decay = torch.exp(log_decay)
                k = (k * decay).to(v.dtype)

            elif self.update_net_type == "dd_decay":
                decay = self.decay_proj(token)
                decay_h = self.decay_h_proj(token)
                h = decay_h.shape[-1]
                # b h n d
                decay = torch.einsum("b n d, b n h -> b h n d", decay, decay_h)
                log_decay = F.logsigmoid(decay)
                # kv = lambda_ * kv + ki * vi
                # 1, a(n-1), a(n-2), ..., a2
                zero = torch.zeros(
                    (b, h, 1, log_decay.shape[-1]), device=torch.cuda.current_device()
                )
                log_decay_reverse = torch.cat(
                    [zero, torch.flip(log_decay, dims=[-2])[:, :, :-1]], dim=-2
                )
                log_decay_cum = torch.cumsum(log_decay_reverse.float(), dim=-2)
                k = (k * torch.exp(log_decay_cum)).to(v.dtype)
            elif self.update_net_type == "dd_share_decay":
                decay_h = self.decay_h_proj(token)
                h = decay_h.shape[-1]
                # b n h -> b h n -> b h n 1
                decay = decay_h.transpose(-1, -2).unsqueeze(-1)
                log_decay = F.logsigmoid(decay)
                # kv = lambda_ * kv + ki * vi
                # 1, a(n-1), a(n-2), ..., a2
                zero = torch.zeros((b, h, 1, 1), device=torch.cuda.current_device())
                log_decay_reverse = torch.cat(
                    [zero, torch.flip(log_decay, dims=[-2])[:, :, :-1]], dim=-2
                )
                log_decay_cum = torch.cumsum(log_decay_reverse.float(), dim=-2)
                k = (k * torch.exp(log_decay_cum)).to(v.dtype)

            if self.update_net_use_conv:
                weight_matrix = torch.einsum("b h n d, b h n e -> b h d e", k, v)
                weight_matrix = self.conv(weight_matrix)
                weight_matrix = rearrange(weight_matrix, "b d n m -> b (n m) d")
            else:
                weight_matrix = torch.einsum("b h n d, b h n e -> b d e h", k, v)
                weight_matrix = rearrange(weight_matrix, "b n m d -> b (n m) d")
        elif self.update_net_type == "delta":
            k = self.act(k)
            k = F.normalize(k, dim=-1)
            v = self.act(v)
            # b n h -> b h n 1
            beta = F.sigmoid(self.beta_proj(token))
            h = beta.shape[-1]
            e = k.shape[-1]
            beta = beta.transpose(-1, -2).unsqueeze(-1)
            S = torch.zeros(b, h, e, e, dtype=torch.float32).to(v)

            for i in range(n):
                _k = k[:, :, i]
                _v = v[:, :, i].clone()
                beta_i = beta[:, :, i]
                _v = _v - (S.clone() * _k[..., None]).sum(-2)
                _v = _v * beta_i
                S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)

            if self.update_net_use_conv:
                weight_matrix = self.conv(S)
                weight_matrix = rearrange(weight_matrix, "b d n m -> b (n m) d")
            else:
                weight_matrix = rearrange(S, "b d n m -> b (n m) d")

        return weight_matrix


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


class Downsample(nn.Module):
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        res = F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return res


# class ResConvAttentionLayer(nn.Module):
#     def __init__(self, cfg, ch_in, ch_out, use_downsample=True):
#         super().__init__()
#         # get params start
#         embed_dim = cfg.hidden_channels
#         num_heads = cfg.num_heads
#         norm_type = cfg.norm_type
#         channel_act = cfg.channel_act
#         use_lrpe = cfg.use_lrpe
#         lrpe_type = cfg.lrpe_type
#         base = cfg.theta_base
#         causal = cfg.causal
#         mid_dim = cfg.mid_dim
#         token_mixer = cfg.token_mixer
#         channel_mixer = cfg.channel_mixer
#         bias = cfg.bias
#         norm_type = cfg.norm_type
#         num_res_blocks = cfg.num_res_blocks
#         # get params end

#         blocks = []
#         for _ in range(num_res_blocks):
#             blocks.append(ResBlock(ch_in, ch_out, bias))
#             ch_in = ch_out

#         self.blocks = nn.Sequential(*blocks)
#         self.proj = nn.Linear(ch_in, ch_out, bias=bias)
#         self.use_downsample = use_downsample
#         if self.use_downsample:
#             self.downsample = Downsample()

#         self.token_mixer = AUTO_TOKEN_MIXER_MAPPING[token_mixer](
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             bias=bias,
#             use_lrpe=use_lrpe,
#             lrpe_type=lrpe_type,
#             base=base,
#             causal=causal,
#             norm_type=norm_type,
#         )

#     def forward(self, x, token):
#         x = self.blocks(x)
#         b, c, h, w = x.shape
#         if self.use_downsample:
#             x = self.downsample(x)

#         n = token.shape[1]
#         token = self.proj(token)
#         x = rearrange(x, "b c h w -> b (h w) c")
#         print(token.shape, x.shape)
#         input = torch.cat([token, x], dim=1)
#         output = self.token_mixer(input)
#         token, x = output[:, :n], output[:, n:]
#         x = rearrange(x, "b (h w) c -> b c h w", h=h)

#         return token, x


# class TransformerResConvEncoder(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         # get params start
#         channels = cfg.hidden_channels
#         num_res_blocks = cfg.num_res_blocks
#         channel_multipliers = cfg.channel_multipliers
#         embed_dim = cfg.embed_dim
#         bias = cfg.bias
#         num_extra_token = cfg.num_extra_token
#         norm_type = cfg.norm_type
#         # get params end

#         self.conv_in = nn.Conv2d(
#             3, channels, kernel_size=3, stride=1, padding=1, bias=bias
#         )

#         self.num_extra_token = num_extra_token
#         self.extra_token = nn.Parameter(torch.randn(num_extra_token, channels))

#         blocks = []
#         ch_in = channels

#         for i in range(len(channel_multipliers)):
#             ch_out = channels * channel_multipliers[i]
#             blocks.append(ResConvAttentionLayer(cfg, ch_in, ch_out))
#             ch_in = ch_out

#         self.blocks = nn.ModuleList(blocks)

#         self.final_blocks = nn.ModuleList(
#             [ResConvAttentionLayer(cfg, ch_in, ch_in, use_downsample=False) for i in range(num_res_blocks)]
#         )

#         self.final_norm = AUTO_NORM_MAPPING[norm_type](ch_in)
#         self.out_proj = nn.Linear(ch_in, embed_dim, bias=bias)

#     def forward(self, x):
#         x = self.conv_in(x)
#         token = repeat(self.extra_token, "n d -> b n d", b=x.shape[0])
#         for i in range(len(self.blocks)):
#             x, token = self.blocks[i](x, token)

#         # b n d -> b n e
#         token = self.out_proj(self.final_norm(token))

#         return token


class ResConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        channels = cfg.hidden_channels
        num_res_blocks = cfg.num_res_blocks
        channel_multipliers = cfg.channel_multipliers
        cfg.embed_dim
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

        self.norm = GroupNorm(num_groups=NUM_GROUPS, num_channels=ch_in)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.final_residual(x)
        x = self.norm(x)
        x = F.silu(x)

        return x


class TransformerResConvEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        cfg.image_size
        cfg.patch_size
        cfg.in_channels
        bias = cfg.bias
        use_ape = cfg.use_ape
        embed_dim = cfg.hidden_channels
        out_dim = cfg.embed_dim
        base = cfg.theta_base
        norm_type = cfg.norm_type
        cfg.patch_embed_name
        cfg.dct_block_size
        cfg.use_zigzag
        cfg.use_freq_patch
        init_std = cfg.init_std
        init_method = cfg.init_method
        token_init_method = cfg.token_init_method
        cfg.patch_merge_size
        cfg.use_channel_pe
        sample_step = cfg.sample_step
        num_extra_token = cfg.num_extra_token
        token_pe_type = cfg.token_pe_type
        token_first = cfg.token_first
        mask_ratio = cfg.mask_ratio
        channel_multipliers = cfg.channel_multipliers
        # get params end

        self.resnet = ResConvEncoder(cfg)
        ch_out = embed_dim * channel_multipliers[-1]
        self.proj = nn.Linear(ch_out, embed_dim, bias=bias)
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

        self.sample_step = sample_step
        self.init_std = init_std
        self.token_first = token_first
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio  # only concat support this

        self.initialize_weights(init_method, token_init_method)

    def initialize_weights(self, init_method, token_init_method):
        self.apply(AUTO_INIT_MAPPING[init_method])
        AUTO_TOKEN_INIT_MAPPING[token_init_method](
            self.extra_token, std=self.embed_dim**-0.5
        )  # std only use when init_method="titok"

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
        x = self.resnet(x)
        input_shape = x.shape[-2:]
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)

        num_patch = x.shape[1]

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

                if self.training and self.mask_ratio:
                    if self.token_first:
                        x, token = x[:, self.sample_step :], x[:, : self.sample_step]
                    else:
                        x, token = x[:, : self.sample_step], x[:, self.sample_step :]

                    x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
                    num_patch = x.shape[1]

                    if self.token_first:
                        x = torch.cat((token, x), dim=1)
                    else:
                        x = torch.cat((x, token), dim=1)

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
            x = layer(x, input_shape)

        if self.token_first:
            # [m n] -> [m]
            x = x[:, :-num_patch]
        else:
            # [n m] -> [m]
            x = x[:, num_patch:]

        # b n d -> b n e
        x = self.out_proj(self.final_norm(x))

        return x


class TransformerResConvDecoder(nn.Module):
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
        init_std = cfg.init_std
        init_method = cfg.init_method
        cfg.causal = cfg.decoder_causal
        cfg.update_net_version
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

        cfg.num_h_patch = self.reverse_patch_embed.num_h_patch
        cfg.num_w_patch = self.reverse_patch_embed.num_w_patch
        self.update_net = UpdateNetV2(cfg)

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

        x = self.reverse_patch_embed(self.final_norm(x))

        return x
