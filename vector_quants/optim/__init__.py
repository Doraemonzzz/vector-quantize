import logging

import torch.optim as optim

from vector_quants.utils import logging_info

OPTIM_DICT = {"adamw": optim.AdamW, "adam": optim.Adam}


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            logging_info(f"no decay: {name}")
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_opt_args(cfg_train):
    opt_kwargs = {"lr": cfg_train.lr, "weight_decay": cfg_train.weight_decay}
    optimizer_name = cfg_train.optimizer_name

    if optimizer_name in ["adam", "adamw"]:
        opt_kwargs["betas"] = (cfg_train.adam_beta1, cfg_train.adam_beta2)
        opt_kwargs["eps"] = cfg_train.adam_epsilon

    return opt_kwargs


def get_optimizer(cfg_train, model):
    weight_decay = cfg_train.weight_decay
    optimizer_name = cfg_train.optimizer_name
    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    parameters = add_weight_decay(model, weight_decay, skip)

    opt_kwargs = get_opt_args(cfg_train)
    optimizer = OPTIM_DICT[optimizer_name](parameters, **opt_kwargs)

    return optimizer
