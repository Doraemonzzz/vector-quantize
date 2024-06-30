import torch
from einops import rearrange
from torch import nn

from vector_quants.quantizers import get_quantizer


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channel = args.in_channel
        channel = args.channel
        embed_dim = args.embed_dim

        blocks = [
            nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
        ]

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel = args.embed_dim
        out_channel = args.in_channel
        channel = args.channel

        blocks = [
            nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1),
        ]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend(
            [
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_channel, 1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.quantizer = get_quantizer(args)

        self.enc = Encoder(args)
        self.dec = Decoder(args)

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
        if self.args.quantizer == "ema" or self.args.quantizer == "origin":
            quant_t, diff_t, id_t = self.quantizer(logits)
            diff_t = diff_t.unsqueeze(0)
            loss_dict = {"codebook_loss": diff_t}
        elif self.args.quantizer in ["fsq", "sfsq"]:
            quant_t, id_t = self.quantizer(logits)
            diff_t = torch.tensor(0.0).cuda().float()
            loss_dict = {"codebook_loss": diff_t}
        elif self.args.quantizer == "lfq":
            # quantized, indices, entropy_aux_loss = quantizer(image_feats)
            quant_t, id_t, diff_t = self.quantizer(logits)
            loss_dict = {"codebook_loss": diff_t}
        elif self.args.quantizer in [
            "rvq",
            "Vq",
            "EmaVq",
            "GumbelVq",
            "SoftmaxVq",
            "Gvq",
            "Hvq",
            "Cvq",
            "Rvq",
            "Lfq",
            "Fsq",
            "Raq",
            "Rfsq",
        ]:

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
