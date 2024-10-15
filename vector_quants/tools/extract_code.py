# Modified from: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_c2i.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
from dataclasses import asdict
from pprint import pformat

import numpy as np
import torch.distributed as dist
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.models import AutoVqVae
from vector_quants.utils import get_cfg, logging_info, set_random_seed, type_dict


def main():
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # get config
    cfg = get_cfg()

    logging_info(pformat(asdict(cfg)))
    cfg.model
    cfg_model_stage2 = cfg.model_stage2
    cfg_train = cfg.train
    cfg_data = cfg.data
    cfg.loss
    cfg.sample

    # setup
    distributed.enable(overwrite=True)
    rank = dist.get_rank()
    rank % torch.cuda.device_count()
    set_random_seed(cfg_train.seed)

    # Setup a feature folder:
    train_code_path = os.path.join(
        cfg_data.code_path, f"{cfg_data.data_set}{cfg_data.image_size}_train"
    )
    if rank == 0:
        os.makedirs(cfg_data.code_path, exist_ok=True)
        os.makedirs(f"{train_code_path}_codes", exist_ok=True)
        os.makedirs(f"{train_code_path}_labels", exist_ok=True)

    # create and load model
    vqvae, vqvae_config, res = AutoVqVae.from_pretrained(
        cfg_train.ckpt_path_stage1,
        embed_dim_stage1=cfg_model_stage2.embed_dim_stage1,
    )
    vqvae.cuda(torch.cuda.current_device())
    vqvae.eval()
    logging_info(vqvae)
    dtype = type_dict[cfg_train.dtype]

    # Setup data:
    train_data_loader = get_data_loaders(cfg_train, cfg_data, is_train=True)

    logging_info("Extracting codes from traning dataset.")
    total = 0
    for input_img, input_label in tqdm(train_data_loader):
        input_img = input_img.cuda(torch.cuda.current_device())
        class_idx = input_label.cuda(torch.cuda.current_device())
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            with torch.no_grad():
                idx = vqvae.img_to_indice(
                    input_img,
                    use_group_id=True,  # for quantizer has group concept(e.g, Fsq, Rvq, Gvq), we save the code as (n g)
                )
        x = idx.detach().cpu().numpy()  # (b, n)
        y = class_idx.detach().cpu().numpy()  # (b,)

        train_steps = rank + total
        np.save(f"{train_code_path}_codes/{train_steps}.npy", x)
        np.save(f"{train_code_path}_labels/{train_steps}.npy", y)
        total += dist.get_world_size()


if __name__ == "__main__":
    main()
