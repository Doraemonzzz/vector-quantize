from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from xmixers.modules import get_norm_fn

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None


class ClassEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_class, embed_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob >= 0
        self.embedding_table = nn.Embedding(num_class + use_cfg_embedding, embed_dim)
        self.num_class = num_class
        self.dropout_prob = dropout_prob

    def token_drop(self, labels):
        """
        Drops labels to enable classifier-free guidance.
        """
        drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        labels = torch.where(drop_ids, self.num_class, labels)
        return labels

    def forward(self, labels):
        use_dropout = self.dropout_prob > 0
        if self.training and use_dropout:
            labels = self.token_drop(labels)

        embeddings = self.embedding_table(labels).unsqueeze(1)

        return embeddings


class TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        embed_dim = cfg.embed_dim
        num_heads = cfg.num_heads
        norm_type = cfg.norm_type
        channel_act = cfg.channel_act
        use_lrpe = cfg.use_lrpe
        lrpe_type = cfg.lrpe_type
        base = cfg.theta_base
        causal = cfg.causal
        mid_dim = cfg.mid_dim
        token_mixer = "softmax_ar"
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

    def forward(self, x, past_key_value=None, shape=None):
        residual = x
        x, past_key_value_new = self.token_mixer(
            self.token_norm(x), past_key_value=past_key_value, shape=shape
        )
        x = x + residual
        x = x + self.channel_mixer(self.channel_norm(x))

        return x, past_key_value_new


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        vocab_size = cfg.vocab_size
        num_class = cfg.num_class
        embed_dim = cfg.embed_dim
        class_dropout_prob = cfg.class_dropout_prob
        num_layers = cfg.num_layers
        norm_type = cfg.norm_type
        bias = cfg.bias
        base = cfg.theta_base
        use_ape = cfg.use_ape
        vocab_groups = cfg.vocab_groups
        tie_word_embeddings = cfg.tie_word_embeddings
        num_group_mixing_layer = cfg.num_group_mixing_layer
        # get params end

        # setup group
        if len(vocab_groups) == 1:
            assert (
                vocab_groups[0] == -1 or vocab_groups[0] == vocab_size
            ), "vocab_groups[0] must be -1 or vocab_size when len(vocab_groups) == 1"
            vocab_groups[0] = vocab_size
        else:
            # vocab_size_group = torch.prod(torch.tensor(vocab_groups)).item()
            vocab_size_group = reduce(lambda x, y: x * y, vocab_groups)
            assert (
                vocab_size_group == vocab_size
            ), f"prod(vocab_groups): {vocab_size_group} must be vocab_size: {vocab_size}"

        _levels = torch.tensor(vocab_groups, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + vocab_groups[:-1]), dim=0, dtype=torch.int64
        )
        self.register_buffer("_basis", _basis, persistent=False)
        self.vocab_groups = vocab_groups
        self.vocab_size_proc = torch.sum(torch.tensor(vocab_groups)).item()

        self.cfg = cfg
        self.codebook_size = vocab_size
        self.num_class = num_class
        self.tie_word_embeddings = tie_word_embeddings

        num_embed = vocab_size + num_class + 1
        self.num_embed = num_embed

        # construct embedding start
        token_embed = nn.ModuleList()
        for vocab_size_ in vocab_groups:
            token_embed.append(nn.Embedding(vocab_size_, embed_dim))
        self.token_embed = nn.ModuleList(token_embed)

        self.class_embed = ClassEmbedder(
            num_class,
            embed_dim,
            class_dropout_prob,
        )
        # construct embedding end

        cfg.causal = True
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for layer_idx in range(num_layers)]
        )
        self.final_norm = get_norm_fn(norm_type)(embed_dim)
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(embed_dim, self.vocab_size_proc, bias=bias)

        self.num_group_mixing_layer = num_group_mixing_layer
        if self.num_group_mixing_layer > 0:
            cfg.causal = False
            # b n (g d) -> (b n) g d
            self.group_proj_in = nn.Linear(
                embed_dim, embed_dim * len(vocab_groups), bias=bias
            )
            self.group_mixing_layers = nn.ModuleList(
                [
                    TransformerLayer(cfg)
                    for layer_idx in range(self.num_group_mixing_layer)
                ]
            )

        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        if not self.tie_word_embeddings:
            nn.init.constant_(self.lm_head.weight, 0)

    def _init_weights(self, module):
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward_embed(self, x, embed_type=0):
        if embed_type == 0:
            if len(x.shape) == 2:
                code = (x.unsqueeze(-1) // self._basis) % self._levels
            else:
                code = x
            output = 0
            for i, embed in enumerate(self.token_embed):
                output = output + embed(code[..., i])
        elif embed_type == 1:
            output = self.class_embed(x)

        return output

    def forward_lmhead(self, x):
        if self.tie_word_embeddings:
            output_list = []
            for i, embed in enumerate(self.token_embed):
                output_list.append(F.linear(x, embed.weight))
            output = torch.cat(output_list, dim=-1)
        else:
            output = self.lm_head(x)

        return output

    def get_lm_head_weights(self):
        if self.tie_word_embeddings:
            output = []
            for i, embed in enumerate(self.token_embed):
                output.append(embed.weight)
        else:
            output = self.lm_head.weight.split(self._levels.tolist(), dim=0)

        return output

    def forward(
        self,
        idx=None,
        cond_idx=None,
        past_key_values=None,
        shape=None,
        target=None,
    ):
        # compute embed
        if idx is not None and cond_idx is not None:  # training
            cond_embed = self.forward_embed(cond_idx, embed_type=1)
            token_embed = self.forward_embed(idx, embed_type=0)
            hidden_state = torch.cat(
                [
                    cond_embed,
                    token_embed,
                ],
                dim=-2,
            )
        elif cond_idx is not None:  # prefill
            hidden_state = self.forward_embed(cond_idx, embed_type=1)
        else:  # decode
            hidden_state = self.forward_embed(idx, embed_type=0)

        offset = 0
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        else:
            offset = past_key_values[0][0].shape[-2]

        new_past_key_values = [None] * len(self.layers)

        if self.use_ape:
            hidden_state = self.pe(hidden_state, offset=offset, shape=shape)

        # (b, *)
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_state,
                past_key_value=past_key_values[i],
                shape=shape,
            )
            hidden_state = layer_outputs[0]
            new_past_key_values[i] = layer_outputs[1]

        if self.num_group_mixing_layer > 0:
            b, n, d = hidden_state.shape
            hidden_state = self.group_proj_in(hidden_state)
            hidden_state = rearrange(
                hidden_state, "b n (g d) -> (b n) g d", g=len(self.vocab_groups)
            )
            for layer in self.group_mixing_layers:
                layer_outputs = layer(
                    hidden_state,
                    past_key_value=None,
                )
                hidden_state = layer_outputs[0]
            hidden_state = torch.mean(hidden_state, dim=-2)
            hidden_state = rearrange(hidden_state, "(b n) d -> b n d", b=b)

        hidden_state = self.final_norm(hidden_state)

        if LigerFusedLinearCrossEntropyLoss is not None and self.training:
            loss_fn = LigerFusedLinearCrossEntropyLoss()
            dtype = (
                torch.get_autocast_gpu_dtype()
                if torch.is_autocast_enabled()
                else hidden_state.dtype
            )
            logits = None
            loss_list = []
            hidden_state = hidden_state[:, :-1]
            lm_head_weights = self.get_lm_head_weights()
            target = (target.unsqueeze(-1) // self._basis) % self._levels
            for i, lm_head_weight in enumerate(lm_head_weights):
                loss_list.append(
                    loss_fn(
                        lm_head_weight.contiguous().to(dtype),
                        hidden_state.contiguous()
                        .view(-1, hidden_state.shape[-1])
                        .to(dtype),
                        target[..., i].long().contiguous().view(-1),
                    ),
                )

            loss = torch.mean(torch.stack(loss_list, dim=0))
        else:
            logits = self.forward_lmhead(hidden_state)
            loss = 0

            if self.training:
                logits = logits[:, :-1]
                logits = logits.split(self._levels.tolist(), dim=-1)
                loss_list = []
                target = (target.unsqueeze(-1) // self._basis) % self._levels
                for i, logits_i in enumerate(logits):
                    loss_list.append(
                        F.cross_entropy(
                            logits_i.contiguous().view(-1, logits_i.shape[-1]),
                            target[..., i].long().contiguous().view(-1),
                        )
                    )
                loss = torch.mean(torch.stack(loss_list, dim=0))

        return logits, new_past_key_values, loss

    def top_k_logits(self, logits_list, k):
        output = []
        for logits in logits_list:
            k = min(k, logits.shape[-1])
            # only sample topk
            v, ix = torch.topk(logits, k, dim=-1)
            out = logits.clone()
            out[out < v[..., [-1]]] = -float("inf")
            output.append(out)

        return output

    @torch.no_grad()
    def generate(
        self,
        steps,
        c=None,
        cfg_scale=1.0,
        cfg_scheduler=None,
        temperature=1.0,
        top_k=None,
    ):
        self.eval()
        # always use cfg, when cfg_scale = 1, we get condition generation
        cond_null = torch.ones_like(c) * self.num_class
        c = torch.cat([c, cond_null], dim=0)

        shape = [steps]
        # prefill
        past_key_values = None
        start = 0
        x = None

        for k in range(start, steps):
            cfg_scale = (
                cfg_scheduler.get_cfg() if cfg_scheduler is not None else cfg_scale
            )

            cond_idx = c if k == 0 else None
            logits, past_key_values, _ = self.forward(
                idx=x, cond_idx=cond_idx, past_key_values=past_key_values, shape=shape
            )

            # get the last token's logits
            # b V
            logits = (
                logits[
                    :,
                    -1,
                ]
                / temperature
            )
            logits, logits_uncond = logits.chunk(2, dim=0)
            logits = logits_uncond + cfg_scale * (logits - logits_uncond)

            # split over group
            logits_list = logits.split(self._levels.tolist(), dim=-1)

            if top_k is not None:
                logits_list = self.top_k_logits(logits_list, top_k)

            idx_list = []
            for logits in logits_list:
                probs = F.softmax(logits, dim=-1)
                # probs: b n v
                x = torch.multinomial(probs, num_samples=1)
                idx_list.append(x)
            idx_new = (
                (torch.cat(idx_list, dim=-1) * self._basis).sum(dim=-1).to(torch.int64)
            )
            x = idx_new.unsqueeze(-1)
            idx = torch.cat([idx, x], dim=1) if k != 0 else x

            x = torch.cat([x, x], dim=0)

        del past_key_values

        return idx
