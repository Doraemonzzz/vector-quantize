from einops import rearrange
from torch import nn

from vector_quants.quantizers import get_quantizer

from ..backbone import (
    BaselineConvDecoder,
    BaselineConvEncoder,
    BasicConvDecoder,
    BasicConvEncoder,
    ResConvDecoder,
    ResConvEncoder,
    TransformerDecoder,
    TransformerEncoder,
)

AUTO_ENCODER_MAPPING = {
    "baseline_conv": BaselineConvEncoder,
    "basic_conv": BasicConvEncoder,
    "res_conv": ResConvEncoder,
    "transformer": TransformerEncoder,
}
AUTO_DECODER_MAPPING = {
    "baseline_conv": BaselineConvDecoder,
    "basic_conv": BasicConvDecoder,
    "res_conv": ResConvDecoder,
    "transformer": TransformerDecoder,
}


class VqVae(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_name = cfg.model_name

        self.is_conv = "conv" in model_name

        self.quantizer = get_quantizer(cfg)

        self.encoder = AUTO_ENCODER_MAPPING[model_name](cfg)
        self.decoder = AUTO_DECODER_MAPPING[model_name](cfg)

        assert self.cfg.quantizer in [
            "Vq",
            "EmaVq",
            "GumbelVq",
            "Gvq",
            "Hvq",
            "Cvq",
            "Rvq",
            "Lfq",
            "Fsq",
            "Raq",
            "Rfsq",
        ], f"quantizer {self.cfg.quantizer} does not support!"

    @property
    def num_embed(self):
        if hasattr(self.quantizer, "num_embed"):
            return self.quantizer.num_embed
        else:
            return -1

    def forward(self, x):
        x_quant, indice, loss_dict = self.encode(x)
        x_recon = self.decode(x_quant)

        return x_recon, indice, loss_dict

    def encode(self, x):
        logits = self.encoder(x)

        if self.is_conv:
            logits = rearrange(logits, "b c h w -> b h w c")

        # update this later? when evaluation, we does not need loss_dict
        quant_logits, indice, loss_dict = self.quantizer(logits)
        if self.is_conv:
            quant_logits = rearrange(quant_logits, "b h w c -> b c h w")

        return quant_logits, indice, loss_dict

    def decode(self, x_quant):
        output = self.decoder(x_quant)

        return output

    def latent_to_indice(self, x):
        logits = self.encoder(x)
        if self.is_conv:
            logits = rearrange(logits, "b c h w -> b h w c")

        indice = self.quantizer.latent_to_indice(logits)

        return indice
