import torch

from .baseline import VQVAE  # baseline version, only for test
from .vqvae import VqVae
from .vqvae_llamagen import VqVaeLlamaGen

AUTO_VQVAE_MAPPING = {
    "baseline": VQVAE,
    "baseline_conv": VqVae,
    "basic_conv": VqVae,
    "res_conv": VqVae,
    "transformer": VqVae,
    "freq_transformer": VqVae,
    "feature_transformer": VqVae,
    "llamagen": VqVaeLlamaGen,
    "block_dct_transformer": VqVae,
    "feature_dct_transformer": VqVae,
    "spatial_feature_transformer": VqVae,
    "gmlp": VqVae,
    "wm_transformer": VqVae,
}


def get_state_dict(path):
    pkg = torch.load(path, map_location="cpu")
    assert (
        "cfg" in pkg or "model_cfg" in pkg
    ), "At least one of cfg or model_cfg must be included in the ckpt."

    if hasattr(pkg, "model_cfg") or (isinstance(pkg, dict) and "model_cfg" in pkg):
        config = pkg["model_cfg"]
    else:
        config = pkg["cfg"].model

    model_state_dict = pkg["model_state_dict"]

    return config, model_state_dict


class AutoVqVae:
    @classmethod
    def from_config(cls, vqvae_config, **kwargs):
        model_name = (
            vqvae_config.model_name
            if hasattr(vqvae_config, "model_name")
            else vqvae_config["model_name"]
        )

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
        if kwargs["embed_dim_stage1"] != -1:  # add this to avoid early bug
            if not isinstance(
                vqvae_config, dict
            ):  # add this to compatible with llamagen
                vqvae_config.embed_dim = kwargs["embed_dim_stage1"]
        model = cls.from_config(vqvae_config)

        res = model.load_state_dict(model_state_dict)

        return model, vqvae_config, res
