from torch import nn

from ..backbone import TransformerModel

AUTO_AR_MAPPING = {"transformer": TransformerModel}


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


class AutoArModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.model = self.from_config(config)

    def forward(self, x, y=None):
        return self.model(x, y)

    def sample(
        self,
    ):
        pass

    @classmethod
    def from_config(cls, ar_config, **kwargs):
        model_name = ar_config.model_name

        if model_name not in AUTO_AR_MAPPING.keys():
            raise ValueError(
                f"Unknown vector quantization type, got {model_name} - supported types are:"
                f" {list(AUTO_AR_MAPPING.keys())}"
            )

        target_cls = AUTO_AR_MAPPING[model_name]
        return target_cls(ar_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # get state dict
        ar_config, model_state_dict = get_state_dict(pretrained_model_name_or_path)
        model = cls.from_config(ar_config)
        model.load_state_dict(model_state_dict)

        return model
