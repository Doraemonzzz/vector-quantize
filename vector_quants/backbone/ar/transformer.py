import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from xmixers.modules import get_norm_fn

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)


class ClassEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_class, embed_dim, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
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
        causal = True
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
        token_embed_type = cfg.token_embed_type
        embed_dim = cfg.embed_dim
        class_dropout_prob = cfg.class_dropout_prob
        num_layers = cfg.num_layers
        norm_type = cfg.norm_type
        bias = cfg.bias
        base = cfg.theta_base
        use_ape = cfg.use_ape
        num_group = cfg.num_group
        # get params end

        self.cfg = cfg
        self.codebook_size = vocab_size
        self.num_class = num_class

        num_embed = vocab_size + num_class + 1
        self.num_embed = num_embed

        # construct embedding start
        self.token_embed_type = token_embed_type
        self.token_embed = nn.Sequential(
            nn.Embedding(
                vocab_size,
                embed_dim // num_group,
            ),
            Rearrange("... g d -> ... (g d)"),
        )

        self.class_embed = ClassEmbedder(
            num_class,
            embed_dim,
            class_dropout_prob,
        )
        # construct embedding end

        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for layer_idx in range(num_layers)]
        )
        self.final_norm = get_norm_fn(norm_type)(embed_dim)
        self.lm_head = nn.Sequential(
            Rearrange("... (g d) -> ... g d", g=num_group),
            nn.Linear(embed_dim // num_group, vocab_size, bias=bias),
        )

        self.use_ape = use_ape
        if self.use_ape:
            self.pe = SinCosPe(
                embed_dim=embed_dim,
                base=base,
            )

        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # # Zero-out output layers:
        # nn.init.constant_(self.lm_head[1].weight, 0)

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
            output = self.token_embed(x)
        elif embed_type == 1:
            output = self.class_embed(x)

        return output

    def forward(
        self,
        idx=None,
        cond_idx=None,
        past_key_values=None,
        shape=None,
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
            )
            hidden_state = layer_outputs[0]
            new_past_key_values[i] = layer_outputs[1]

        hidden_state = self.final_norm(hidden_state)

        logits = self.lm_head(hidden_state)

        if self.training:
            logits = logits[:, :-1]

        return logits, new_past_key_values
