import os

import torch
from torchvision import datasets, transforms

from .constants import get_mean_std_from_dataset_name
from .indice_dataset import IndiceDataset


def get_transform(args):
    # Train and Val share the same transform
    imagenet_transform = [
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(*get_mean_std_from_dataset_name(args.data_set)),
    ]
    return transforms.Compose(imagenet_transform)


def get_data_loaders_by_args(args, is_train=True):
    transform = get_transform(args)
    if args.data_set == "cifar100":
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
    elif args.data_set == "imagenet-1k":
        if is_train:
            name = "train"
        else:
            name = "val"
        dataset = datasets.ImageFolder(
            os.path.join(args.data_path, name), transform=transform
        )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=is_train,
        seed=args.seed,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=is_train,
    )

    return data_loader


def get_data_loaders(cfg_train, cfg_data, is_train=True, is_indice=False):
    transform = get_transform(cfg_data)
    if is_indice:
        # for generation only
        assert not is_train, "indice dataset should be used for training"
        dataset = IndiceDataset(
            cfg_data,
        )
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
