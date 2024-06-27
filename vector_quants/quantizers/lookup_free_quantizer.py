import torch
import torch.nn.functional as F

from .base_vector_quantizer import BaseVectorQuantizer
from .utils import pack_one, unpack_one


class LookUpFreeQuantizer(BaseVectorQuantizer):
    def __init__(self, cfg):
        super().__init__()
        # get params start
        base = 2
        codebook_value = cfg.codebook_value
        embed_dim = cfg.embed_dim
        use_norm = cfg.use_norm
        commitment_loss_weight = cfg.commitment_loss_weight
        entropy_loss_weight = cfg.entropy_loss_weight
        entropy_temperature = cfg.entropy_temperature
        entropy_loss_type = cfg.entropy_loss_type
        # get params end

        # construct base and levels
        self.base = base
        num_levels = embed_dim
        levels = [base] * num_levels
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=False)

        # other params
        self._num_embed = self._levels.prod().item()
        self.num_levels = self._levels.shape[0]
        self.embed_dim = embed_dim
        self.codebook_value = codebook_value
        self.commitment_loss_weight = commitment_loss_weight
        self.use_norm = use_norm
        self.entropy_temperature = entropy_temperature
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_type = entropy_loss_type

        # init codebook
        self.init_codebook()

    def extra_repr(self):
        return f"(num embedding): {self.num_embed}\n(embed size): {self.embed_dim}"

    @property
    def num_embed(self):
        return self._num_embed

    def init_codebook(self):
        codebook = self.indice_to_code(torch.arange(self.num_embed))
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x):
        # add normalize
        if self.use_norm:
            x = F.normalize(x, dim=-1)

        x_quant, indice = self.latent_to_code_and_indice(x)

        diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x)

        if self.entropy_loss_weight > 0:
            self.entropy_loss_weight * self.entropy_loss(x)
        else:
            torch.tensor(0.0).cuda().float()

        x_quant = x + (x_quant - x).detach()

        return x_quant, diff, indice

    def latent_to_code_and_indice(self, latent):
        mask = latent > 0
        indice = (mask.int() * self._basis).sum(dim=-1).to(torch.int32)
        code = torch.where(mask, self.codebook_value, -self.codebook_value)

        return code, indice

    def latent_to_indice(self, latent):
        # (b, *, d) -> (n, d)
        latent, ps = pack_one(latent, "* d")
        indice = ((latent > 0).int() * self._basis).sum(dim=-1).to(torch.int32)

        indice = unpack_one(indice, ps, "*")

        return indice

    def indice_to_code(self, indice):
        # (..., d)
        indice = (indice.unsqueeze(-1) // self._basis) % self._levels
        code = torch.where(indice > 0, self.codebook_value, -self.codebook_value)

        return code
