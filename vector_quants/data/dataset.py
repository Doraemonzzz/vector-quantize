import os

import torch
from torchvision import datasets, transforms

from .augment import new_data_aug_generator
from .code_dataset import CodeDataset
from .constants import get_mean_std_from_dataset_name
from .indice_dataset import IndiceDataset


def get_transform(cfg_data, cfg_train, is_train=True):
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

        if is_train:
            transform = [
                transforms.Resize(cfg_data.image_size),
                transforms.RandomCrop(cfg_data.image_size),
                transforms.RandomHorizontalFlip(),
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
    cfg_train, cfg_data, is_train=True, is_indice=False, use_pre_tokenize=False
):
    transform = get_transform(cfg_data, cfg_train, is_train=is_train)
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
