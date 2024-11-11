# https://github.com/rakkit/any_diffusion

import pickle
from pathlib import Path

import torch

try:
    from any_diffusion.autoencoders.flexTokenizer.flexTokenizer import FlexTokenizer
except ImportError:
    FlexTokenizer = object


class VqVaeAnyDiffusion(FlexTokenizer):
    def __init__(self, path):
        super().__init__()
        self.model = FlexTokenizer.init_and_load_from(path)
        if hasattr(self, "encoder"):
            del self.encoder
        if hasattr(self, "decoder"):
            del self.decoder
        if hasattr(self, "quantizers"):
            del self.quantizers
        # load config
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location="cpu")
        config = pickle.loads(pkg["config"])
        self.codebook_size = config["quantizer_config"]["num_embed"]
        self.num_group = config["quantizer_config"]["num_group"]

    def img_to_indice(self, x, **kwargs):
        indice = self.model.tokenize(x)

        return indice

    def indice_to_img(self, x, **kwargs):
        quantized = self.model.indice_to_code(x)
        output = self.model.decoder(quantized)

        return output

    @property
    def num_embed(self):
        return self.codebook_size**self.num_group
