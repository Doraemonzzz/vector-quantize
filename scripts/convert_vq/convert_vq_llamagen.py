import argparse
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


def main(args):
    ckpt_dir = args.ckpt_dir
    ckpt_name = args.ckpt_name
    ckpt = f"{ckpt_dir}/{ckpt_name}.pt"

    vq = torch.load(ckpt, map_location="cpu")

    if "ds16" in ckpt_name:
        ch_mult = [1, 1, 2, 2, 4]
    else:
        ch_mult = [1, 2, 2, 4]

    torch.save(
        {
            "model_state_dict": vq["model"],
            "model_cfg": {
                "model_name": "llamagen",
                "encoder_ch_mult": ch_mult,
                "decoder_ch_mult": ch_mult,
                "codebook_size": 16384,
                "codebook_embed_dim": 8,
            },
        },
        f"{ckpt_dir}/{ckpt_name}_proc.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
