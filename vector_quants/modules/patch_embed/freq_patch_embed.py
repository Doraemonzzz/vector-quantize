from einops import rearrange
from torch import nn

from vector_quants.ops import dct_2d, idct_2d, zigzag_indices
from vector_quants.utils import pair


class FreqPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        flatten=True,
        bias=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.flatten = flatten
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch

        self.to_patch_embedding = nn.Linear(
            channels * self.num_patch, embed_dim, bias=bias
        )
        indices, reverse_indices = zigzag_indices(
            self.patch_embed.num_h_patch, self.patch_embed.num_w_patch
        )
        self.register_buffer("indices", indices, persistent=False)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b h w c p1 p2",
            h=self.num_h_patch,
            w=self.num_w_patch,
        )
        y = dct_2d(x)
        y = rearrange(y, "b h w c p1 p2 -> b h w (c p1 p2)")
        y = self.to_patch_embedding(x)
        y = rearrange(y, "b h w c -> b (h w) c")

        return y


class ReversePatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        flatten=True,
        bias=False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.flatten = flatten
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch

        self.reverse_patch_embedding = nn.Linear(
            embed_dim, channels * self.num_patch, bias=bias
        )
        indices, reverse_indices = zigzag_indices(
            self.reverse_patch_embed.num_h_patch, self.reverse_patch_embed.num_w_patch
        )
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)

    def forward(self, x):
        if self.flatten:
            x = rearrange(x, "b (h w) c -> b h w c", h=self.num_h_patch)

        y = self.reverse_patch_embedding(x)
        y = rearrange(
            y,
            "b h w (c p1 p2) -> b h w c p1 p2",
            h=self.num_h_patch,
            w=self.num_w_patch,
        )
        y = idct_2d(y)

        y = self.reverse_patch_embedding(x.contiguous())

        return y
