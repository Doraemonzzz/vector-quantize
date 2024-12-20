from einops import rearrange
from torch import nn

from vector_quants.backbone import (
    BaselineConvDecoder,
    BaselineConvEncoder,
    BasicConvDecoder,
    BasicConvEncoder,
    BlockDctTransformerDecoder,
    BlockDctTransformerEncoder,
    FeatureDctTransformerDecoder,
    FeatureDctTransformerEncoder,
    FeatureTransformerDecoder,
    FeatureTransformerEncoder,
    FreqTransformerDecoder,
    FreqTransformerEncoder,
    GMlpDecoder,
    GMlpEncoder,
    ResConvDecoder,
    ResConvEncoder,
    SFTransformerDecoder,
    SFTransformerEncoder,
    TransformerDecoder,
    TransformerEncoder,
    TransformerResConvDecoder,
    TransformerResConvEncoder,
    UpdateNet,
    WeightMatrixTransformerDecoder,
    WeightMatrixTransformerDecoderV2,
    WeightMatrixTransformerEncoder,
    WeightMatrixTransformerEncoderV2,
    WMTCDecoder,
    WMTCEncoder,
)
from vector_quants.quantizers import QUANTIZER_DICT, get_quantizer

AUTO_ENCODER_MAPPING = {
    "baseline_conv": BaselineConvEncoder,
    "basic_conv": BasicConvEncoder,
    "res_conv": ResConvEncoder,
    "transformer": TransformerEncoder,
    "freq_transformer": FreqTransformerEncoder,
    "feature_transformer": FeatureTransformerEncoder,
    "block_dct_transformer": BlockDctTransformerEncoder,
    "feature_dct_transformer": FeatureDctTransformerEncoder,
    "spatial_feature_transformer": SFTransformerEncoder,
    "gmlp": GMlpEncoder,
    "wm_transformer": WeightMatrixTransformerEncoder,
    "wm_transformer_v2": WeightMatrixTransformerEncoderV2,
    "wmtc": WMTCEncoder,
    "transformer_resnet": TransformerResConvEncoder,
}

AUTO_DECODER_MAPPING = {
    "baseline_conv": BaselineConvDecoder,
    "basic_conv": BasicConvDecoder,
    "res_conv": ResConvDecoder,
    "transformer": TransformerDecoder,
    "freq_transformer": FreqTransformerDecoder,
    "feature_transformer": FeatureTransformerDecoder,
    "block_dct_transformer": BlockDctTransformerDecoder,
    "feature_dct_transformer": FeatureDctTransformerDecoder,
    "spatial_feature_transformer": SFTransformerDecoder,
    "gmlp": GMlpDecoder,
    "wm_transformer": WeightMatrixTransformerDecoder,
    "wm_transformer_v2": WeightMatrixTransformerDecoderV2,
    "wmtc": WMTCDecoder,
    "transformer_resnet": TransformerResConvDecoder,
}


class VqVae(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = cfg.model_name

        self.is_conv = "conv" in self.model_name

        self.quant_spatial = cfg.quant_spatial

        self.encoder = AUTO_ENCODER_MAPPING[self.model_name](cfg)
        self.decoder = AUTO_DECODER_MAPPING[self.model_name](cfg)

        if self.model_name in ["wm_transformer"]:
            self.update_net = UpdateNet(cfg)

        assert self.cfg.quantizer in list(
            QUANTIZER_DICT.keys()
        ), f"quantizer {self.cfg.quantizer} does not support!"

        if self.is_conv and self.quant_spatial:
            assert False, "quant_spatial does not support conv now!"

        if self.quant_spatial:
            origin_embed_dim = cfg.embed_dim
            cfg.embed_dim = self.encoder.num_patch

        self.quantizer = get_quantizer(cfg)
        if (
            self.quant_spatial
        ):  # change this back, other wise it will cause bug in stage2
            cfg.embed_dim = origin_embed_dim

    @property
    def num_embed(self):
        if hasattr(self.quantizer, "num_embed"):
            return self.quantizer.num_embed
        else:
            return -1

    def forward(self, x, step=-1):
        x_quant, indice, loss_dict = self.encode(x)
        x_recon = self.decode(x_quant, step=step)

        return x_recon, indice, loss_dict

    def encode(self, x):
        logits = self.encoder(x)
        if self.is_conv:
            logits = rearrange(logits, "b c h w -> b h w c")

        if self.quant_spatial:
            logits = rearrange(logits, "b n c -> b c n")
        # update this later? when evaluation, we does not need loss_dict
        if self.model_name in ["wm_transformer"]:
            logits, target_logits = (
                logits[:, -self.encoder.sample_step :],
                logits[:, : -self.encoder.sample_step],
            )

        quant_logits, indice, loss_dict = self.quantizer(logits)

        if self.quant_spatial:
            quant_logits = rearrange(quant_logits, "b c n -> b n c")
        if self.is_conv:
            quant_logits = rearrange(quant_logits, "b h w c -> b c h w")

        if self.model_name in ["wm_transformer"]:
            quant_logits, wm_l1_loss = self.update_net(quant_logits, target_logits)
            loss_dict.update({"wm_l1_loss": wm_l1_loss})

        return quant_logits, indice, loss_dict

    def decode(self, x_quant, step=-1):
        output = self.decoder(x_quant, step=step)

        return output

    def img_to_indice(self, x):
        logits = self.encoder(x)
        if self.is_conv:
            logits = rearrange(logits, "b c h w -> b h w c")

        if self.quant_spatial:
            logits = rearrange(logits, "b n c -> b c n")

        indice = self.quantizer.latent_to_indice(logits)

        return indice

    def indice_to_img(self, x):
        quant_logits = self.quantizer.indice_to_code(x)
        if self.quant_spatial:
            quant_logits = rearrange(quant_logits, "b c n -> b n c")
        if self.is_conv:
            quant_logits = rearrange(quant_logits, "b h w c -> b c h w")

        output = self.decoder(quant_logits)

        return output
