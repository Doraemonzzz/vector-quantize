import os
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf
from simple_parsing import ArgumentParser

from .utils import get_num_embed


@dataclass
class ModelConfig:
    in_channel: int = field(default=3, metadata={"help": "Input Channel"})
    channel: int = field(default=512, metadata={"help": "Middle Channel"})
    quantizer: str = field(
        default="ema",
        metadata={
            "help": "Use which quantizer",
            "choices": ["ema", "origin", "fsq", "sfsq", "lfq"],
        },
    )
    levels: List[int] = field(
        default_factory=lambda: [8, 5, 5, 5], metadata={"help": "FSQ levels"}
    )
    lfq_dim: int = field(default=10, metadata={"help": "Look up free quantizer dim"})
    embed_dim: int = field(
        default=256, metadata={"help": "The embedding dimension of VQVAE's codebook"}
    )
    n_embed: int = field(
        default=1024, metadata={"help": "The number embeddings of VQVAE's codebook"}
    )
    model_name: str = field(default="baseline", metadata={"help": "Model name"})


@dataclass
class TrainingConfig:
    experiment_name: str = field(
        default="VQVAE",
        metadata={"help": "The experiment name for summary and checkpoint"},
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size per gpu"})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    save_interval: int = field(
        default=10, metadata={"help": "Number of epochs between saves"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay coefficient for L2 regularization"}
    )
    clip_grad: float = field(default=0.0, metadata={"help": "Gradient clipping"})
    train_iters: int = field(
        default=500000,
        metadata={"help": "Total number of iterations to train over all training runs"},
    )
    max_train_epochs: int = field(
        default=100,
        metadata={"help": "Total number of epochs to train over all training runs"},
    )
    log_interval: int = field(default=100, metadata={"help": "Report interval"})
    exit_interval: Optional[int] = field(
        default=None,
        metadata={"help": "Exit the program after this many new iterations."},
    )
    seed: int = field(default=1234, metadata={"help": "Random seed"})
    lr_decay_style: str = field(
        default="linear",
        metadata={
            "help": "Learning rate decay function",
            "choices": ["constant", "linear", "cosine", "exponential"],
        },
    )
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "LR decay ratio"})
    lr: float = field(default=1.0e-4, metadata={"help": "Initial learning rate"})
    warmup: float = field(
        default=0.01, metadata={"help": "Percentage of data to warmup on"}
    )
    save: Optional[str] = field(
        default=None, metadata={"help": "Output directory to save checkpoints to."}
    )
    recon_save: bool = field(
        default=True, metadata={"help": "Whether to save reconstruction sample"}
    )
    load: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a directory containing a model checkpoint."},
    )
    no_load_optim: bool = field(
        default=False,
        metadata={"help": "Do not load optimizer when loading checkpoint."},
    )
    distributed_backend: str = field(
        default="nccl",
        metadata={"help": "Which backend to use for distributed training."},
    )
    local_rank: Optional[int] = field(
        default=None, metadata={"help": "Local rank passed from distributed launcher"}
    )
    dist_url: str = field(
        default="env://", metadata={"help": "URL used to set up distributed training"}
    )
    train_modules: str = field(default="all", metadata={"help": "Module to train"})
    dtype: str = field(
        default="fp32",
        metadata={"help": "Training dtype", "choices": ["fp32", "fp16", "bf16"]},
    )
    output_dir: str = field(default="checkpoints", metadata={"help": "Output dir"})
    output_name: Optional[str] = field(default=None, metadata={"help": "Output name"})
    loss_type: int = field(default=1, metadata={"help": "Loss type"})
    ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint path"}
    )
    eval_only: bool = field(default=False, metadata={"help": "Evaluation only"})
    eval_interval: int = field(default=5, metadata={"help": "Evaluation interval"})
    use_wandb: bool = field(default=False, metadata={"help": "Use wandb"})
    wandb_entity: str = field(default="", metadata={"help": "wandb entity"})
    wandb_project: str = field(default="", metadata={"help": "wandb project"})
    wandb_exp_name: str = field(default="", metadata={"help": "wandb experiment name"})
    wandb_cache_dir: str = field(
        default="wandb", metadata={"help": "wandb cache directory"}
    )
    log_freq: int = field(default=100, metadata={"help": "Log frequency"})
    # optimizer
    optimizer_name: str = field(default="adam", metadata={"help": "Optimizer name"})
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for Adam optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
    )


@dataclass
class DataConfig:
    data_set: str = field(
        default="imagenet-1k",
        metadata={"help": "Dataset path", "choices": ["cifar100", "imagenet-1k"]},
    )
    data_path: str = field(default="./images/", metadata={"help": "Dataset path"})
    img_size: int = field(default=128, metadata={"help": "Size of image for dataset"})
    num_workers: int = field(
        default=8, metadata={"help": "Number of workers to use for dataloading"}
    )


@dataclass
class LossConfig:
    fid_feature: int = field(default=2048, metadata={"help": "FID feature"})
    perceptual_loss_type: int = field(
        default=1, metadata={"help": "Perceptual loss type"}
    )
    adversarial_loss_type: int = field(
        default=0, metadata={"help": "Adversarial loss type"}
    )
    post_transform_type: int = field(
        default=1, metadata={"help": "Post transform type before compute loss"}
    )
    entropy_loss_weight: float = field(
        default=0.0, metadata={"help": "Entropy loss weight"}
    )
    commitment_loss_weight: float = field(
        default=0.0, metadata={"help": "Commitment loss weight"}
    )
    l1_loss_weight: float = field(default=1.0, metadata={"help": "L1 loss weight"})
    l2_loss_weight: float = field(default=0.0, metadata={"help": "L2 loss weight"})
    perceptual_loss_weight: float = field(
        default=1.0, metadata={"help": "Perceptual loss weight"}
    )
    adversarial_loss_weight: float = field(
        default=0.0, metadata={"help": "Adversarial loss weight"}
    )
    codebook_loss_weight: float = field(
        default=1.0, metadata={"help": "Codebook loss weight"}
    )
    metrics_list: str = field(default="", metadata={"help": "Evaluation Metrics list"})


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()
    cfg_path: str = field(default="", metadata={"help": "Config path"})


def merge_config(yaml_config, args):
    # Create OmegaConf from dataclass
    config = OmegaConf.structured(Config())

    # Merge YAML config if provided
    if yaml_config:
        config = OmegaConf.merge(config, yaml_config)

    # Convert argparse Namespace to dictionary and create OmegaConf
    args_dict = vars(args)
    cli_config = {k: v for k, v in args_dict.items() if v is not None}
    cli_config = OmegaConf.create(cli_config)

    # Merge CLI config
    config = OmegaConf.merge(config, cli_config)

    # Convert OmegaConf to dataclass instance
    return OmegaConf.to_object(config)


def process_args(args):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    total_batch_size = args.train.batch_size * world_size

    output_name = args.train.output_name
    if output_name is None:
        output_name = f"{args.model.quantizer}-is{args.data.img_size}-bs{total_batch_size}-lr{args.train.lr}-wd{args.train.weight_decay}"
    save = f"{args.train.output_dir}/{output_name}"
    postfix = ""
    quantizer = args.model.quantizer

    if quantizer == "lfq":
        args.model.embed_dim = args.model.lfq_dim
        postfix = f"-entropy_weights-{args.loss.entropy_loss_weight}-codebook_weights-{args.loss.codebook_loss_weight}-lfq_dim-{args.model.lfq_dim}"

    elif quantizer in ["fsq", "sfsq"]:
        args.model.embed_dim = len(args.model.levels)
        postfix = f"-n_embed-{get_num_embed(args.model)}"

    else:
        postfix = f"-n_embed-{get_num_embed(args.model)}"

    args.train.save = save + postfix

    return args


def get_cfg():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="args")
    args = parser.parse_args().args
    args = process_args(args)

    yaml_config = OmegaConf.create()
    if args.cfg_path:
        yaml_config = OmegaConf.load(args.cfg_path)
    config = merge_config(yaml_config, args)

    return config
