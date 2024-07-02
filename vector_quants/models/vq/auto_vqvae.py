import torch

from .baseline import VQVAE  # baseline version, only for test
from .vqvae import VqVae

AUTO_VQVAE_MAPPING = {
    "baseline": VQVAE,
    "baseline_conv": VqVae,
    "basic_conv": VqVae,
    "res_conv": VqVae,
}


def get_state_dict(path):
    pkg = torch.load(path, map_location="cpu")
    assert (
        "cfg" in pkg or "model_cfg" in pkg
    ), "At least one of cfg or model_cfg must be included in the ckpt."
    if hasattr(pkg, "model_cfg"):
        config = pkg["model_cfg"]
    else:
        config = pkg["cfg"].model

    model_state_dict = pkg["model_state_dict"]

    return config, model_state_dict


class AutoVqVae:
    @classmethod
    def from_config(cls, vqvae_config, **kwargs):
        model_name = vqvae_config.model_name

        if model_name not in AUTO_VQVAE_MAPPING.keys():
            raise ValueError(
                f"Unknown vector quantization type, got {model_name} - supported types are:"
                f" {list(AUTO_VQVAE_MAPPING.keys())}"
            )

        target_cls = AUTO_VQVAE_MAPPING[model_name]
        return target_cls(vqvae_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # get state dict
        vqvae_config, model_state_dict = get_state_dict(pretrained_model_name_or_path)
        model = cls.from_config(vqvae_config)
        model.load_state_dict(model_state_dict)

        return model
