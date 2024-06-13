# Credit to https://github.com/rakkit 
DATASET_CONFIGS = {
    "mnist": {
        "image_size": 28,
        "mean": (0.1307, 0.1307, 0.1307),
        "std": (0.3081, 0.3081, 0.3081),
        "samples": 60000,
        "data_keys": ["image", "label"],
    },
    "cifar10": {
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "samples": 50000,
        "data_keys": ["img", "label"],
    },
    "cifar100": {
        "image_size": 32,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "samples": 50000,
        "data_keys": ["img", "fine_label"],
    },
    "imagenet-1k": {
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "samples": 1281167,
        "data_keys": ["image", "label"],
    },
    "celeba-hq": {
        "image_size": 256,
        "mean": (0.5, 0.5, 0.5),
        "std": (1, 1, 1),
        "data_keys": ["image", "label"],
    },
}


def get_mean_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]["mean"]
    else:
        return [0, 0, 0]


def get_std_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]["std"]
    else:
        return [1, 1, 1]


def get_mean_std_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    mean, std = get_mean_from_dataset_name(dataset_name), get_std_from_dataset_name(
        dataset_name
    )
    return mean, std
