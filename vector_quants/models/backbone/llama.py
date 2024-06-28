import torch
import torch.nn as nn
from einops import repeat
from xmixers.models import LlamaLayer
from xmixers.modules import get_norm_fn

from vector_quants.quantizers.utils import pack_one, unpack_one


class LlamaModel(nn.Module):
    def __init__(self, config):
        # [0, codebook_size - 1]:                         quantized image token
        # [codebook_size, codebook_size + num_class - 1]: class token
        # codebook_size + num_class:                      drop condition
        super().__init__()
        self.codebook_size = config.vocab_size
        self.num_class = config.num_class

        num_embed = config.vocab_size + config.num_class + 1
        self.num_embed = num_embed

        self.embed_tokens = nn.Embedding(
            num_embed,
            config.embed_dim,
        )
        self.layers = nn.ModuleList(
            [LlamaLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.final_norm = get_norm_fn(config.norm_type)(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)

    def forward(
        self,
        x,
        y=None,
        past_key_values=None,
    ):
        b = x.shape[0]
        if y == None:
            y = repeat(torch.tensor([self.num_embed - 1]).cuda(), "... -> b ...", b=b)

        # (b, *)
        x, ps = pack_one(x, "b *")
        token = torch.cat([y, x], dim=-1)
        hidden_states = self.embed_tokens(token)
        for idx, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                past_key_values=past_key_values,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)[:, :-1]
        logits = unpack_one(logits, ps, "b * d")

        return logits
