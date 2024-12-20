import argparse
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf
from simple_parsing import ArgumentParser


@dataclass
class ModelConfig:
    in_channels: int = field(default=3, metadata={"help": "Input Channel"})
    hidden_channels: int = field(default=512, metadata={"help": "Hidden Channel"})
    hidden_channels_wt: int = field(
        default=512, metadata={"help": "Hidden Channel use in weight_matrix_tc"}
    )
    quantizer: str = field(
        default="ema",
        metadata={
            "help": "Use which quantizer",
            "choices": [
                "Vq",
                "EmaVq",
                "GumbelVq",
                "Gvq",
                "Gsvq",
                "Hvq",
                "Cvq",
                "Rvq",
                "Lfq",
                "Fsq",
                "Raq",
                "Rfsq",
                "Rcq",
                "Ivq",
            ],
        },
    )
    vq_init_type: str = field(
        default="uniform",
        metadata={
            "help": "Codebook init type in vq",
            "choices": ["uniform", "normal", "data"],
        },
    )
    vq_norm_type: str = field(default="none", metadata={"help": "Norm type in vq"})
    levels: List[int] = field(
        default_factory=lambda: [8, 5, 5, 5], metadata={"help": "FSQ levels"}
    )
    base: int = field(default=8, metadata={"help": "Base for CVQ,RAQ"})
    num_levels: int = field(default=4, metadata={"help": "Number of levels for CVQ"})
    lfq_dim: int = field(default=10, metadata={"help": "Look up free quantizer dim"})
    embed_dim: int = field(
        default=256, metadata={"help": "The embedding dimension of VQVAE's codebook"}
    )
    num_embed: int = field(
        default=1024, metadata={"help": "The number embeddings of VQVAE's codebook"}
    )
    num_group: int = field(
        default=8, metadata={"help": "The number group of VQVAE's codebook"}
    )
    num_residual: int = field(
        default=1, metadata={"help": "The number residual of VQVAE's codebook"}
    )
    num_patch: int = field(
        default=1, metadata={"help": "Number of patches to the quantizer"}
    )
    model_name: str = field(
        default="baseline",
        metadata={
            "help": "Model name",
            "choices": [
                "baseline",
                "baseline_conv",
                "basic_conv",
                "res_conv",
                "transformer",
                "freq_transformer",
                "feature_transformer",
                "block_dct_transformer",
                "feature_dct_transformer",
                "spatial_feature_transformer",
                "gmlp",
                "wm_transformer",
                "wm_transformer_v2",
                "wmtc",
                "transformer_resnet",
            ],
        },
    )
    patch_embed_name: str = field(
        default="vanilla",
        metadata={
            "help": "Patch embed name",
            "choices": ["vanilla", "freq", "block_dct"],
        },
    )
    quant_spatial: bool = field(
        default=False, metadata={"help": "Whether use spatial quantization"}
    )
    # loss weight
    commitment_loss_weight: float = field(
        default=1.0, metadata={"help": "Commitment loss weight"}
    )
    kl_loss_weight: float = field(default=5e-4, metadata={"help": "KL loss weight"})
    sample_entropy_loss_weight: float = field(
        default=0.0, metadata={"help": "Sample entropy loss weight"}
    )
    codebook_entropy_loss_weight: float = field(
        default=0.0, metadata={"help": "Codebook entropy loss weight"}
    )
    # entropy loss
    entropy_loss_weight: float = field(
        default=0.0, metadata={"help": "Entropy loss weight"}
    )
    entropy_temperature: float = field(
        default=1.0,
        metadata={"help": "Non-negative scalar temperature for entropy loss"},
    )
    entropy_loss_type: str = field(
        default="softmax",
        metadata={
            "help": "Use which entropy",
            "choices": ["softmax", "argmax"],
        },
    )
    # for lfq
    codebook_value: float = field(
        default=1.0, metadata={"help": "Codebook value for lfq"}
    )
    # for rq
    ema_decay: float = field(default=0.99, metadata={"help": "Ema decay for codebook"})
    shared_codebook: bool = field(
        default=False,
        metadata={
            "help": "If True, codebooks are shared in all location. If False, uses separate codebooks along the depth dimension. (default: False)"
        },
    )
    # for gumbel vq
    kl_temperature: float = field(
        default=1.0,
        metadata={"help": "Non-negative scalar temperature for gumbel softmax"},
    )
    straight_through: bool = field(
        default=True,
        metadata={
            "help": "if True, will one-hot quantize, but still differentiate as if it is the soft sample"
        },
    )
    ##### backbone start
    # resnet
    num_conv_blocks: int = field(
        default=2,
        metadata={"help": "Number of conv blocks in BasicConvEncoder/Decoder"},
    )
    num_res_blocks: int = field(
        default=2,
        metadata={
            "help": "Number of residual blocks in every stage of ResConvEncoder/Decoder"
        },
    )
    channel_multipliers: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8],
        metadata={"help": "channel multipliers of ResConvEncoder/Decoder"},
    )
    # transformer
    use_ape: bool = field(
        default=True,
        metadata={"help": "Whether to use ape in transformer or linear transformer."},
    )
    theta_base: int = field(
        default=10000, metadata={"help": "Theta base for SinCosPe and Lrpe"}
    )
    num_extra_token: int = field(
        default=0, metadata={"help": "Number extra token used in transformer"}
    )
    norm_type: str = field(
        default="layernorm",
        metadata={
            "help": "Normalization type",
            "choices": ["layernorm", "simplermsnorm"],
        },
    )
    num_layers: int = field(
        default=12, metadata={"help": "The number of layers of transformer model"}
    )
    num_heads: int = field(
        default=4, metadata={"help": "Number of heads in attention/linear attention"}
    )
    patch_size: int = field(
        default=14, metadata={"help": "Patch size of transformer or linear transformer"}
    )
    channel_act: str = field(
        default="silu",
        metadata={
            "help": "Channel Mixer Activation function type",
            "choices": [
                "silu",
            ],
        },
    )
    use_lrpe: bool = field(default=False, metadata={"help": "Whether use lrpe or not"})
    lrpe_type: int = field(
        default=1, metadata={"help": "Lrpe type for attentin/linear attention"}
    )
    causal: bool = field(
        default=False, metadata={"help": "Whether use causal attention or not"}
    )
    mid_dim: int = field(
        default=512, metadata={"help": "The mid dimension of stage2 model"}
    )
    token_mixer: str = field(
        default="softmax",
        metadata={
            "help": "Token Mixer type",
            "choices": ["softmax", "linear_attention", "gmlp"],
        },
    )
    channel_mixer: str = field(
        default="glu",
        metadata={
            "help": "Channel Mixer type",
            "choices": [
                "ffn",
                "glu",
            ],
        },
    )
    ##### backbone end
    # others
    use_norm: bool = field(
        default=True, metadata={"help": "Whether to use normalize in Quantizer"}
    )
    bias: bool = field(
        default=False, metadata={"help": "Whether use bias in nn.linear or not"}
    )
    dct_block_size: int = field(default=-1, metadata={"help": "DCT block size"})
    use_zigzag: bool = field(
        default=False, metadata={"help": "Whether use zigzag order or not"}
    )
    use_freq_patch: bool = field(
        default=False, metadata={"help": "Whether use freq patch or not"}
    )
    transpose_feature: bool = field(
        default=False,
        metadata={"help": "Whether transpose feature in encoder/decoder or not"},
    )
    use_block_dct_only: bool = field(
        default=False, metadata={"help": "Whether use block dct only or not in encoder"}
    )
    use_feature_pooling: bool = field(
        default=True,
        metadata={
            "help": "Whether use feature pooling or not in feature dct transformer"
        },
    )
    init_std: float = field(
        default=0.02, metadata={"help": "The std for initialization"}
    )
    use_init: bool = field(
        default=False,
        metadata={"help": "Whether use special initialization in transformer or not"},
    )
    init_method: str = field(
        default="no_init",
        metadata={
            "help": "Initialization method",
            "choices": ["timm", "vit_jax", "llama_gen", "fla", "fairseq", "no_init"],
        },
    )
    token_init_method: str = field(
        default="no_init",
        metadata={
            "help": "Token initialization method",
            "choices": ["titok", "vit_vqgan_jax", "mae", "timm", "no_init"],
        },
    )
    use_rescale: bool = field(
        default=False,
        metadata={
            "help": "Whether to use rescale on out projection",
        },
    )
    patch_merge_size: int = field(
        default=1,
        metadata={
            "help": "Whether merge patch use overlap patch embedding in encoder/decoder or not, i.e. (b n d) -> (b m 1)"
        },
    )
    num_feature_layers: int = field(
        default=1,
        metadata={"help": "The number of layers of feature transformer model"},
    )
    use_channel_pe: bool = field(
        default=False,
        metadata={"help": "Whether use channel pe or not in transformer"},
    )
    update_net_version: int = field(
        default=1,
        metadata={"help": "The version of update net"},
    )
    update_net_type: str = field(
        default="additive",
        metadata={
            "help": "Model name",
            "choices": [
                "additive",
                "di_decay",  # data independent decay
                "dd_decay",  # data dependent decay
                "dd_share_decay",  # data dependent decay with head share
                "additive_decay",
                "cosine",
                "rope",
                "delta",
            ],
        },
    )
    update_net_act: str = field(
        default="silu",
        metadata={
            "help": "Update net activation function type",
        },
    )
    update_net_use_conv: bool = field(
        default=False,
        metadata={
            "help": "Whether use conv in update net or not",
        },
    )
    token_pe_type: str = field(
        default="concat",
        metadata={
            "help": "Whether use token pe type in weight matrix transformer",
            "choices": [
                "concat",  # concat to the origin token, then pe
                "sincos",  # no concat, sincos
                "learnable",  # no concat, learnable
            ],
        },
    )
    decoder_causal: bool = field(
        default=False,
        metadata={"help": "Whether use causal attention or not in decoder"},
    )
    token_first: bool = field(
        default=False,
        metadata={
            "help": "Whether use concat[token, x] or concat[x, token] in wm transformer"
        },
    )
    mask_ratio: float = field(default=0, metadata={"help": "Mask ratio for encoder"})
    frozen_blocks: str = field(
        default_factory=lambda: "",
        metadata={"help": "frozen blocks, split by comma"},
    )
    ema: bool = field(default=False, metadata={"help": "Whether use EMA model"})


@dataclass
class ModelStage2Config:
    model_name: str = field(
        default="transformer",
        metadata={
            "help": "Model name for stage2",
            "choices": ["transformer", "transformer_llamagen"],
        },
    )
    vocab_size: int = field(default=1024, metadata={"help": "Size of codebook"})
    embed_dim: int = field(
        default=512, metadata={"help": "The embedding dimension of stage2 model"}
    )
    mid_dim: int = field(
        default=512, metadata={"help": "The mid dimension of stage2 model"}
    )
    num_layers: int = field(
        default=12, metadata={"help": "The number of layers of stage2 model"}
    )
    num_heads: int = field(
        default=4, metadata={"help": "Number of heads in attention/linear attention"}
    )
    kv_heads: int = field(
        default=-1,
        metadata={
            "help": "Number of kv heads in attention/linear attention, only use this in MQA,GQA"
        },
    )
    bias: bool = field(
        default=False, metadata={"help": "Whether use bias in nn.linear or not"}
    )
    use_ape: bool = field(
        default=True,
        metadata={"help": "Whether to use ape in transformer or linear transformer."},
    )
    use_lrpe: bool = field(default=False, metadata={"help": "Whether use lrpe or not"})
    lrpe_type: int = field(
        default=1, metadata={"help": "Lrpe type for attentin/linear attention"}
    )
    theta_base: int = field(
        default=10000, metadata={"help": "Theta base for SinCosPe and Lrpe"}
    )
    norm_type: str = field(
        default="layernorm",
        metadata={
            "help": "Normalization type",
            "choices": ["layernorm", "simplermsnorm"],
        },
    )
    channel_mixer: str = field(
        default="glu",
        metadata={
            "help": "Channel Mixer type",
            "choices": [
                "ffn",
                "glu",
            ],
        },
    )
    channel_act: str = field(
        default="silu",
        metadata={
            "help": "Channel Mixer Activation function type",
            "choices": [
                "silu",
            ],
        },
    )
    token_embedding_type: str = field(
        default="default",
        metadata={
            "help": "Token embedding type",
            "choices": ["default", "group"],
        },
    )
    class_dropout_prob: float = field(
        default=0.1, metadata={"help": "Class dropout probability"}
    )
    init_std: float = field(
        default=0.02, metadata={"help": "The std for initialization"}
    )
    embed_dim_stage1: int = field(
        default=-1, metadata={"help": "The embedding dimension of stage1 model"}
    )
    token_embed_type: str = field(
        default="default",
        metadata={
            "help": "Token embedding type",
            "choices": ["default", "group"],
        },
    )
    use_group_id: bool = field(
        default=False, metadata={"help": "Whether to use group id in stage2 model"}
    )
    vocab_groups: List[int] = field(
        default_factory=lambda: [-1],
        metadata={
            "help": "Vocab groups, for example, if we vocab size is 2^16, we can factorize it as [2^8, 2^8]. Use -1 to denote no group"
        },
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to tie word embeddings with lm head or not"},
    )
    num_group_mixing_layer: int = field(
        default=0,
        metadata={"help": "Number of group mixing layers in group mixing block"},
    )
    ema: bool = field(default=False, metadata={"help": "Whether use EMA model"})


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
    ckpt_path_stage1: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint path"}
    )
    ckpt_path_stage2: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint path of stage2"}
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
    optimizer_name: str = field(
        default="adamw",
        metadata={
            "help": "Optimizer name",
            "choices": [
                "adam",
                "adamw",
            ],
        },
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for Adam optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for Adam optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for Adam optimizer."}
    )
    only_load_model: bool = field(
        default=False,
        metadata={"help": "Only load model without optimizer, scheduler, etc."},
    )
    ref_batch: str = field(
        default="",
        metadata={"help": "Reference batch path for openai evaluation"},
    )
    compile: bool = field(
        default=False,
        metadata={"help": "Whether to use torch.compile"},
    )


@dataclass
class DataConfig:
    data_set: str = field(
        default="imagenet-1k",
        metadata={"help": "Dataset path", "choices": ["cifar100", "imagenet-1k"]},
    )
    data_path: str = field(default="./images/", metadata={"help": "Dataset path"})
    image_size: int = field(default=128, metadata={"help": "Size of image for dataset"})
    num_workers: int = field(
        default=8, metadata={"help": "Number of workers to use for dataloading"}
    )
    # for evaluation
    num_sample: int = field(
        default=50000, metadata={"help": "Number of samples for evaluation"}
    )
    # data augument
    three_augment: bool = field(
        default=False, metadata={"help": "Whether use three augment"}
    )
    src: bool = field(default=False, metadata={"help": "Simple random crop"})
    color_jitter: float = field(
        default=0.3, metadata={"help": "Color jitter factor (default: 0.3)"}
    )
    # pre tokenize params
    code_path: str = field(
        default="",
        metadata={"help": "Path for pre-tokenization"},
    )
    feature_dir: str = field(
        default="",
        metadata={"help": "Path for pre-tokenization feature"},
    )
    label_dir: str = field(
        default="",
        metadata={"help": "Path for pre-tokenization label"},
    )
    use_pre_tokenize: bool = field(
        default=False, metadata={"help": "Whether use pre-tokenization"}
    )
    sample_folder_dir: str = field(
        default="",
        metadata={"help": "Path for sample folder"},
    )
    val_folder_dir: str = field(
        default="",
        metadata={"help": "Path for val folder"},
    )


@dataclass
class LossConfig:
    fid_feature: int = field(default=2048, metadata={"help": "FID feature"})
    perceptual_loss_type: int = field(
        default=1, metadata={"help": "Perceptual loss type"}
    )
    perceptual_model_name: str = field(
        default="resnet18.a1_in1k",
        metadata={
            "help": "Perceptual model name",
            "choices": [
                "vgg16.tv_in1k",
                "resnet18.a1_in1k",
                "resnet34.a2_in1k",
                "resnet50.tv_in1k",
                "resnet152.a3_in1k",
                "deit_tiny_distilled_patch16_224.fb_in1k",
                "deit_small_distilled_patch16_224.fb_in1k",
                "deit_base_distilled_patch16_224.fb_in1k",
                "swin_tiny_patch4_window7_224.ms_in22k",
                "swin_small_patch4_window7_224.ms_in22k",
                "swin_base_patch4_window7_224.ms_in22k",
            ],
        },
    )
    post_transform_type: int = field(
        default=1, metadata={"help": "Post transform type before compute loss"}
    )
    l1_loss_weight: float = field(default=1.0, metadata={"help": "L1 loss weight"})
    l2_loss_weight: float = field(default=0.0, metadata={"help": "L2 loss weight"})
    perceptual_loss_weight: float = field(
        default=1.0, metadata={"help": "Perceptual loss weight"}
    )
    codebook_loss_weight: float = field(
        default=1.0, metadata={"help": "Codebook loss weight"}
    )
    wm_l1_loss_weight: float = field(
        default=0.0, metadata={"help": "Weight Matric loss weight"}
    )
    metrics_list: str = field(default="", metadata={"help": "Evaluation Metrics list"})
    # for discriminator
    disc_loss_start_iter: int = field(
        default=20000, metadata={"help": "Discriminator loss start iteration"}
    )
    disc_type: str = field(
        default="none",
        metadata={
            "help": "Discriminator type",
            "choices": ["patchgan", "stylegan", "dino", "none"],
        },
    )
    gen_loss_type: str = field(
        default="hinge",
        metadata={
            "help": "Generator loss type",
            "choices": ["hinge", "vanilla", "non-saturating"],
        },
    )
    gen_loss_weight: float = field(
        default=0.1, metadata={"help": "Generator loss weight"}
    )
    disc_loss_type: str = field(
        default="hinge",
        metadata={
            "help": "Discriminator loss type",
            "choices": ["hinge", "vanilla", "non-saturating"],
        },
    )
    disc_model_name: str = field(
        default="vit_small_patch16_224.dino",
        metadata={
            "help": "Discriminator model name",
            "choices": [
                "vit_small_patch16_224.dino",
                "vit_small_patch8_224.dino",
                "vit_small_patch14_dinov2.lvd142m",
                "vit_small_patch14_reg4_dinov2.lvd142m",
            ],
        },
    )
    disc_loss_weight: float = field(
        default=1.0, metadata={"help": "Discriminator loss weight"}
    )
    gp_loss_type: str = field(
        default="none",
        metadata={
            "help": "Gradient penalty loss type",
            "choices": ["wgan", "l2", "none"],
        },
    )
    gp_loss_weight: float = field(
        default=0.0, metadata={"help": "Gradient penalty loss weight for discriminator"}
    )
    fid_statistics_file: Optional[str] = field(
        default=None, metadata={"help": "FID statistics file"}
    )


@dataclass
class SampleConfig:
    sample_step: int = field(default=128, metadata={"help": "Number of sample step"})
    cfg_scale_list: List[float] = field(
        default_factory=lambda: [],
        metadata={"help": "Config scale list"},
    )
    cfg_schedule_list: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "Config schedule list"},
    )
    save_npz: bool = field(default=False, metadata={"help": "Whether save npz file"})
    sample_file: bool = field(default=False, metadata={"help": "Whether sample file"})


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainingConfig = TrainingConfig()
    sample: SampleConfig = SampleConfig()
    data: DataConfig = DataConfig()
    loss: LossConfig = LossConfig()
    model_stage2: ModelStage2Config = ModelStage2Config()
    cfg_path: str = field(default="", metadata={"help": "Config path"})


def merge_config(args, yaml_config, cmd_args_dict):
    # Convert argparse Namespace to dictionary and create OmegaConf
    args_dict = vars(args)
    default_config = {k: v for k, v in args_dict.items() if v is not None}
    default_config = OmegaConf.create(default_config)

    OmegaConf.create(cmd_args_dict)

    # Create OmegaConf from dataclass
    config = OmegaConf.structured(Config())
    # merge
    config = OmegaConf.merge(config, default_config)
    config = OmegaConf.merge(config, yaml_config)
    config = OmegaConf.merge(config, cmd_args_dict)

    # Convert OmegaConf to dataclass instance
    return OmegaConf.to_object(config)


def process_args(args):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    total_batch_size = args.train.batch_size * world_size

    output_name = args.train.output_name
    if output_name is None:
        output_name = f"{args.model.quantizer}-is{args.data.image_size}-bs{total_batch_size}-lr{args.train.lr}-wd{args.train.weight_decay}"
    args.train.save = f"{args.train.output_dir}/{output_name}"
    quantizer = args.model.quantizer

    if quantizer in ["Fsq", "Rfsq", "Rcq"] and not args.model.quant_spatial:
        args.model.embed_dim = len(args.model.levels)

    return args


def get_cfg(args_list=sys.argv[1:]):
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="args")
    args = parser.parse_args(args_list).args
    # default args
    args = process_args(args)

    # yaml args
    yaml_config = OmegaConf.create()
    if args.cfg_path:
        yaml_config = OmegaConf.load(args.cfg_path)

    # cmd args
    # credit to https://github.com/pytorch/torchtitan/blob/main/torchtitan/config_manager.py
    # aux parser to parse the command line only args, with no defaults from main parser
    cmd_args_dict = {}
    for name, cfg in vars(args).items():
        if name == "cfg_path":
            cmd_args_dict[name] = cfg
            aux_parser.add_argument("--" + name, type=type(cfg))
        else:
            aux_parser = argparse.ArgumentParser(
                argument_default=argparse.SUPPRESS, allow_abbrev=False
            )  # add this to avoid abbrev bug, such as treat 'lr' as 'lrpe_type'
            for key, val in asdict(cfg).items():
                if isinstance(val, bool):
                    aux_parser.add_argument(
                        "--" + key, action="store_true" if val else "store_false"
                    )
                else:
                    if key in [
                        "levels",
                        "channel_multipliers",
                        "vocab_groups",
                    ]:
                        aux_parser.add_argument("--" + key, type=int, nargs="+")
                    elif key in [
                        "cfg_scale_list",
                    ]:
                        aux_parser.add_argument("--" + key, type=float, nargs="+")
                    elif key in ["cfg_schedule_list"]:
                        aux_parser.add_argument("--" + key, type=str, nargs="+")
                    else:
                        aux_parser.add_argument("--" + key, type=type(val))
            cmd_args, _ = aux_parser.parse_known_args(args_list)

            cmd_args_dict[name] = vars(cmd_args)

    # merge: default -> yaml -> cmd
    config = merge_config(args, yaml_config, cmd_args_dict)

    return config
