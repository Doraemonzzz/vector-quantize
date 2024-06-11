import os

import torch
from torchvision import datasets, transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def get_transform(args):
    # Train and Val share the same transform
    imagenet_transform = [
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
    ]
    return transforms.Compose(imagenet_transform)


def get_data_loaders(args, is_train=True):
    transform = get_transform(args)
    if args.data_set == "CIFAR":
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
    elif args.data_set == "IMNET":
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
