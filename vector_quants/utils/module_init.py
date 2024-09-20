import math

import torch.nn as nn


def init_weights_timm(module):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv1d)
        or isinstance(module, nn.Conv2d)
    ):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_vit_jax(module):
    def is_qkv_merge(module):
        return module.weight.shape[0] == 3 * module.weight.shape[1]

    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if is_qkv_merge(module):
            # treat the weights of Q, K, V separately
            val = math.sqrt(
                6.0 / float(module.weight.shape[0] // 3 + module.weight.shape[1])
            )
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


def init_weights_llama_gen(module):
    """GPT weight initialization, ref llamagen"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def no_init(module):
    return


AUTO_INIT_MAPPING = {
    "timm": init_weights_timm,
    "vit_jax": init_weights_vit_jax,
    "llama_gen": init_weights_llama_gen,
    "no_init": no_init,
}
