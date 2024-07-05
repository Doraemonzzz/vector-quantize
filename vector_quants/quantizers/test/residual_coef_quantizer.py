import torch
import torch.nn as nn
from einops import rearrange

from ..base_vector_quantizer import BaseVectorQuantizer
from ..finite_scalar_quantizer import FiniteScalarQuantizer
from ..utils import pack_one, unpack_one


class ResidualCoefQuantizer(BaseVectorQuantizer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # get params start
        num_residual = cfg.num_residual
        bias = cfg.bias
        embed_dim = cfg.embed_dim
        num_patch = cfg.num_patch
        # get params end

        self.num_residual = num_residual
        cfg.levels = num_patch * cfg.levels
        n = int(num_patch**0.5)
        self.fsq = FiniteScalarQuantizer(cfg)
        self._num_embed = self.fsq.num_embed

        input_size = 3
        hidden_size = 128
        self.net = nn.Sequential(
            nn.LayerNorm(3),
            nn.SiLU(inplace=True),
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.LayerNorm(hidden_size),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_size, embed_dim, bias=bias),
        )

        # build index
        array = [
            torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
            for n in [n, n, num_residual]
        ]
        grid = torch.meshgrid(array)
        # h, w, k, 3
        index = torch.stack(grid, dim=-1)
        self.register_buffer("index", index.float(), persistent=False)

        # init codebook
        self.init_codebook()

    def extra_repr(self):
        return f"(num residual): {self.num_residual}"

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        codebook = self.fsq.indice_to_code(torch.arange(self.num_embed))
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x):
        indice_list = []
        # (b, *, d) -> (b, e)
        x, ps = pack_one(x, "b *")
        x_quant = torch.zeros_like(x)
        residual = x
        bases = rearrange(self.net(self.index), "h w k d -> k (h w d)")

        for i in range(self.num_residual):
            residual_quant, indice, loss_dict = self.fsq(residual)
            base = bases[i]
            # update
            residual = residual - residual_quant.detach() * base.detach()
            x_quant = x_quant + residual_quant * base
            indice_list.append(indice)

        codebook_loss = torch.tensor(0.0).cuda().float()
        loss_dict = {
            "codebook_loss": codebook_loss,
        }

        indice = torch.cat(indice_list, dim=0)

        x_quant = unpack_one(x_quant, ps, "b *")

        return x_quant, indice, loss_dict

    def latent_to_indice(self, latent):
        pass

    def indice_to_code(self, indice):
        pass
