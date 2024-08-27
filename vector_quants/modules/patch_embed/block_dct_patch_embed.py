from einops import rearrange
from torch import nn

from vector_quants.ops import block_dct_2d, zigzag_indices
from vector_quants.utils import pair


class BlockDctPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        dct_block_size,
        embed_dim,
        channels=3,
        bias=False,
        use_zigzag=False,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        self.num_h_patch = dct_block_size
        self.num_w_patch = dct_block_size
        self.num_patch = self.num_h_patch * self.num_w_patch
        self.dct_block_size = dct_block_size

        patch_size = image_size // dct_block_size
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width

        dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Linear(dim, embed_dim, bias=bias)

        self.use_zigzag = use_zigzag
        if self.use_zigzag:
            indices, reverse_indices = zigzag_indices(
                self.dct_block_size, self.dct_block_size
            )
            self.register_buffer("indices", indices, persistent=False)

    def forward(self, x):
        x = block_dct_2d(x, norm="ortho", block_size=self.dct_block_size)

        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w c) (p1 p2)",
            p1=self.dct_block_size,
            p2=self.dct_block_size,
        )
        x = rearrange(x, "b n d -> b d n")

        y = self.to_patch_embedding(x)

        if self.use_zigzag:  # take dct coef as seqlen
            y = y[:, self.indices]

        return y


class ReverseBlockDctPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        dct_block_size=8,
        use_zigzag=False,
        transpose_feature=False,  # if True, use dct coef as seq dim
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        self.transpose_feature = transpose_feature
        if self.transpose_feature:

            self.num_h_patch = dct_block_size
            self.num_w_patch = dct_block_size
            self.num_patch = self.num_h_patch * self.num_w_patch

            patch_size = image_size // dct_block_size

            self.patch_height = patch_size
            self.patch_width = patch_size

            dim = channels * patch_size * patch_size
        else:
            image_height, image_width = pair(image_size)
            patch_height, patch_width = pair(patch_size)

            self.num_h_patch = image_height // patch_height
            self.num_w_patch = image_width // patch_width
            self.num_patch = self.num_h_patch * self.num_w_patch

            self.patch_height = patch_height
            self.patch_width = patch_width

            dim = channels * patch_height * patch_width

        self.dct_block_size = dct_block_size
        self.reverse_patch_embedding = nn.Linear(embed_dim, dim, bias=bias)

    def forward(self, x):

        y = self.reverse_patch_embedding(x)

        if self.transpose_feature:
            y = rearrange(
                y,
                "b (p1 p2) (h w c) -> b c (h p1) (w p2)",
                h=self.patch_height,
                w=self.patch_width,
                p1=self.dct_block_size,
                p2=self.dct_block_size,
            )
        else:
            y = rearrange(
                y,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.num_h_patch,
                w=self.num_w_patch,
                p1=self.patch_height,
                p2=self.patch_width,
            )

        return y
