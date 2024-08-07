import torch
import torch.nn as nn
from einops import repeat
from xmixers.modules import get_norm_fn

from vector_quants.quantizers.utils import pack_one, unpack_one

from vector_quants.modules import (
    AUTO_CHANNEL_MIXER_MAPPING,
    AUTO_NORM_MAPPING,
    AUTO_PATCH_EMBED_MAPPING,
    AUTO_REVERSE_PATCH_EMBED_MAPPING,
    AUTO_TOKEN_MIXER_MAPPING,
    SinCosPe,
)
from vector_quants.utils import print_module


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

    def forward(self, x, past_key_value=None,shape=None):
        residual = x
        x, past_key_value_new = self.token_mixer(self.token_norm(x), past_key_value=past_key_value, shape=shape)
        x = x + residual
        x = x + self.channel_mixer(self.channel_norm(x))

        return x, past_key_value_new

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codebook_size = config.vocab_size
        self.num_class = config.num_class

        num_embed = config.vocab_size + config.num_class + 1
        self.num_embed = num_embed

        # construct embedding start
        self.token_embed_type = config.token_embedding_type
        self.token_embed = self.construct_token_embed()
        # random change condition to null like dit
        self.class_embed = ClassEmbedder(
            config.num_class,
            config.embed_dim,
            config.class_dropout_prob,
        )
        # construct embedding end
        
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for layer_idx in range(config.num_layers)]
        )
        self.final_norm = get_norm_fn(config.norm_type)(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)

    def construct_token_embed(self):
        if self.token_embed_type in ["group"]:
            return nn.Embedding(
                self.config.vocab_size,
                self.config.embed_dim // self.config.num_group,
            )
        else:
            return nn.Embedding(
                self.config.vocab_size,
                self.config.embed_dim,
            )

    def forward_embed(self, x, embed_type=0):
        if embed_type == 0:
            if self.token_embed_type in ["group"]:
                output = self.token_embed(x)
                output = rearrange(output, "... g d -> ... (g d)")
            else:
                output = self.token_embed(x)
        elif embed_type == 1:
            output = self.class_embed(x)
            
        return output

    def forward(
        self,
        idx=None,
        cond_idx=None,
        past_key_values=None,
    ):
        # compute embed
        if idx is not None and cond_idx is not None: # training
            token_embed = self.forward_embed(idx, embed_type=0)
            cond_embed = self.forward_embed(cond_idx, embed_type=1)
            hidden_state = torch.cat([token_embed, cond_embed], dim=-2)
        elif cond_idx is not None: # prefill
            hidden_state = self.forward_embed(cond_idx, embed_type=1)
        else: # decode
            hidden_state = self.forward_embed(idx, embed_type=0)
        
        past_key_values = [None] * len(self.layers)
        new_past_key_values = [None] * len(self.layers)
        
        # (b, *)
        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_state,
                past_key_value=past_key_values[idx],
            )
            hidden_state = layer_outputs[0]
            new_past_key_values[idx] = layer_outputs[1]

        hidden_state = self.final_norm(hidden_state)
        logits = self.lm_head(hidden_state)[:, :-1]

        return logits, new_past_key_values
