import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from .augment import new_data_aug_generator
from .code_dataset import CodeDataset
from .constants import get_mean_std_from_dataset_name
from .indice_dataset import IndiceDataset


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def get_transform(cfg_data, cfg_train, is_train=True, use_data_aug=True):
    if cfg_data.three_augment and is_train:
        pre_transform = new_data_aug_generator(cfg_data)
        post_transform = [
            transforms.ToTensor(),
            transforms.Normalize(*get_mean_std_from_dataset_name(cfg_data.data_set)),
        ]
        transform = pre_transform + post_transform
    else:
        if (
            cfg_train.ckpt_path_stage1 is not None
            and "llamagen" in cfg_train.ckpt_path_stage1
        ):
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean, std = get_mean_std_from_dataset_name(cfg_data.data_set)

        if is_train and use_data_aug:
            transform = [
                transforms.Resize(cfg_data.image_size),
                transforms.RandomCrop(cfg_data.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        else:
            if (
                "llamagen" in cfg_train.ckpt_path_stage1
            ):  # use this setting can repreduct llamagen's result
                transform = [
                    transforms.Lambda(
                        lambda pil_image: center_crop_arr(
                            pil_image, cfg_data.image_size
                        )
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            else:
                transform = [
                    transforms.Resize(cfg_data.image_size),
                    transforms.CenterCrop(cfg_data.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]

    return transforms.Compose(transform)


def get_data_loaders(
    cfg_train,
    cfg_data,
    is_train=True,
    is_indice=False,
    use_pre_tokenize=False,
    use_data_aug=True,
):
    transform = get_transform(
        cfg_data, cfg_train, is_train=is_train, use_data_aug=use_data_aug
    )
    if is_indice:
        # for generation only
        assert not is_train, "indice dataset should not be used for training"
        dataset = IndiceDataset(
            cfg_data,
        )
    else:
        if use_pre_tokenize:
            dataset = CodeDataset(cfg_data)
        else:
            if cfg_data.data_set == "cifar100":
                dataset = datasets.CIFAR100(
                    cfg_data.data_path, train=is_train, transform=transform
                )
            elif cfg_data.data_set == "imagenet-1k":
                if is_train:
                    name = "train"
                else:
                    name = "val"
                dataset = datasets.ImageFolder(
                    os.path.join(cfg_data.data_path, name), transform=transform
                )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=is_train,
        seed=cfg_train.seed,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg_train.batch_size,
        num_workers=cfg_data.num_workers,
        drop_last=is_train,
    )

    return data_loader
