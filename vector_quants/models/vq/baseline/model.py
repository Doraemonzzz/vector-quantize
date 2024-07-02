from einops import rearrange
from torch import nn

from vector_quants.quantizers import get_quantizer


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg.in_channels
        hidden_channels = cfg.hidden_channels
        embed_dim = cfg.embed_dim

        blocks = [
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),
        ]

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(hidden_channels, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        in_channels = cfg.embed_dim
        out_channel = cfg.in_channels
        hidden_channels = cfg.hidden_channels

        blocks = [
            nn.ConvTranspose2d(in_channels, hidden_channels, 4, stride=2, padding=1),
        ]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend(
            [
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    hidden_channels, hidden_channels, 4, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channel, 1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.quantizer = get_quantizer(cfg)

        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)

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

    def forward(self, input, return_id=True):
        (quant_t, id_t, loss_dict) = self.encode_(input)
        dec = self.dec(quant_t)

        if return_id:
            return dec, id_t, loss_dict
        return dec, loss_dict, _

    def encode_(self, input):
        logits = self.enc(input)

        logits = rearrange(logits, "b c h w -> b h w c")
        quant_t, id_t, loss_dict = self.quantizer(logits)
        quant_t = rearrange(quant_t, "b h w c -> b c h w")

        return quant_t, id_t, loss_dict

    def encode(self, input):
        latent = self.enc(input)
        latent = rearrange(latent, "b c h w -> b h w c")
        indice = self.quantizer.latent_to_indice(latent)

        return indice

    def decode(self, indice):
        code = self.quantizer.indice_to_code(indice)
        output = self.dec(code)

        return output
