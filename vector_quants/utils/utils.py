import logging
import os
import random
import socket

import numpy as np
import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dist init")

type_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def mkdir_ckpt_dirs(args):
    if torch.distributed.get_rank() == 0:
        if os.path.exists(args.save):
            print("savedir already here.", args.save)
        else:
            os.makedirs(args.save)
            os.makedirs(args.save + "/ckpts")
            os.makedirs(args.save + "/samples")

        argsDict = args.__dict__
        with open(os.path.join(args.save, "setting.txt"), "w") as f:
            f.writelines("------------------- start -------------------" + "\n")
            for arg, value in argsDict.items():
                f.writelines(arg + " : " + str(value) + "\n")
            f.writelines("------------------- end -------------------" + "\n")


def multiplyList(myList):

    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result * x
    return result


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_ip():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    return ip


def logging_info(string, warning=False):
    if is_main_process():
        if warning:
            logger.warning(string)
        else:
            logger.info(string)


def rescale_image_tensor(img_tensor, mean, std):
    if isinstance(mean, (int, float)):
        mean = [mean] * 3
    if isinstance(std, (int, float)):
        std = [std] * 3
    mean = torch.tensor(mean).view(3, 1, 1).type_as(img_tensor)
    std = torch.tensor(std).view(3, 1, 1).type_as(img_tensor)
    return img_tensor * std + mean


def update_dict(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = dict2[key]
        else:
            dict1[key] += dict2[key]

    return dict1


def reduce_dict(loss_dict, n, prefix=""):
    dist.get_world_size()
    keys = list(loss_dict.keys())
    tensor = torch.tensor(
        [loss_dict[key] / n for key in keys],
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )

    # Reduce the tensors across all devices
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    res = {}
    for i, key in enumerate(keys):
        res[f"{prefix}{key}"] = tensor[i].item()

    return res


def print_dict(res_dict):
    for key in res_dict:
        logging_info(f"{key}: {res_dict[key]}")


def get_metrics_list(metrics_list):
    if metrics_list == "":
        return []
    else:
        return metrics_list.split(",")


def compute_grad_norm(model, norm_type=2, scale=1):
    if hasattr(model, "module"):
        parameters = model.module.parameters()
    else:
        parameters = model.parameters()

    total_norm = 0
    for param in parameters:
        if param.requires_grad and param.grad is not None:
            grad = param.grad.detach().data.float() / scale
            param_norm = grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1 / norm_type)
    return total_norm


def print_module(module):
    named_modules = set()
    for p in module.named_modules():
        named_modules.update([p[0]])
    named_modules = list(named_modules)

    string_repr = ""
    for p in module.named_parameters():
        name = p[0].split(".")[0]
        if name not in named_modules:
            string_repr = (
                string_repr
                + "("
                + name
                + "): "
                + "Tensor("
                + str(tuple(p[1].shape))
                + ", requires_grad="
                + str(p[1].requires_grad)
                + ")\n"
            )

    return string_repr.rstrip("\n")


def print_params(**kwargs) -> None:
    if is_main_process():
        logging_info(f"start print config of {kwargs['__class__']}")
        for key in kwargs:
            if key in ["__class__", "self"]:
                continue
            logging_info(f"{key}: {kwargs[key]}")
        logging_info(f"end print config of {kwargs['__class__']}")


def print_config(config) -> None:
    if is_main_process():
        logging_info(f"start print config of {config['__class__']}")
        for key in config:
            if key in ["__class__", "self"]:
                continue
            logging_info(f"{key}: {config[key]}")
        logging_info(f"end print config of {config['__class__']}")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def compute_num_patch(cfg_model):
    model_name = cfg_model.model_name
    if "transformer" in model_name or "mlp" in model_name:
        num_patch = (cfg_model.image_size // cfg_model.patch_size) ** 2
    else:
        num_patch = 0
        assert False

    return num_patch


def get_token_embed_type(vqvae_config):
    try:
        if vqvae_config.quantizer in ["Cvq"]:
            return "group"
        else:
            return "default"
    except:
        return "default"


def get_num_group(vqvae_config):
    try:
        if vqvae_config.quantizer in ["Cvq"]:
            return vqvae_config.num_levels
        else:
            return 1
    except:
        return 1


def get_is_1d_token(vqvae_config):
    try:
        if vqvae_config.model_name in ["freq_transformer", "feature_transformer"]:
            return True
        else:
            return False
    except:
        return False
