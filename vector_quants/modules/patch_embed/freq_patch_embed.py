from einops import rearrange
from torch import nn

from vector_quants.ops import block_dct_2d, block_idct_2d, zigzag_indices
from vector_quants.utils import pair


class FreqPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        dct_block_size=8,
        use_zigzag=False,
        use_freq_patch=False,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        if use_freq_patch:
            patch_size = image_size // dct_block_size

        patch_height, patch_width = pair(patch_size)

        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch

        self.to_patch_embedding = nn.Linear(
            channels * patch_height * patch_width, embed_dim, bias=bias
        )
        self.use_freq_patch = use_freq_patch
        self.dct_block_size = dct_block_size
        self.use_zigzag = use_zigzag
        if self.use_zigzag:
            indices, reverse_indices = zigzag_indices(
                self.num_h_patch, self.num_w_patch
            )
            self.register_buffer("indices", indices, persistent=False)

    def forward(self, x):
        if self.dct_block_size > 0:
            x = block_dct_2d(x, norm="ortho", block_size=self.dct_block_size)

        if self.use_freq_patch:
            y = rearrange(
                x,
                "b c (p1 h) (p2 w) -> b (h w) (p1 p2 c)",
                h=self.num_h_patch,
                w=self.num_w_patch,
            )
        else:
            y = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                h=self.num_h_patch,
                w=self.num_w_patch,
            )
        y = self.to_patch_embedding(y)

        if self.use_zigzag:
            y = y[:, self.indices]

        return y


class ReverseFreqPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        dct_block_size=8,
        use_zigzag=False,
        use_freq_patch=False,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        if use_freq_patch:
            patch_size = image_size // dct_block_size
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch
        self.image_height = image_height
        self.image_width = image_width

        self.reverse_patch_embedding = nn.Linear(
            embed_dim, channels * patch_height * patch_width, bias=bias
        )
        self.use_freq_patch = use_freq_patch
        self.dct_block_size = dct_block_size
        self.use_zigzag = use_zigzag
        if self.use_zigzag:
            indices, reverse_indices = zigzag_indices(
                self.num_h_patch, self.num_w_patch
            )
            self.register_buffer("reverse_indices", reverse_indices, persistent=False)

    def forward(self, x):
        y = self.reverse_patch_embedding(x)
        if self.use_zigzag:
            y = y[:, self.reverse_indices]

        if self.use_freq_patch:
            y = rearrange(
                y,
                "b (h w) (p1 p2 c) -> b c (p1 h) (p2 w)",
                h=self.num_h_patch,
                w=self.num_w_patch,
                p1=self.patch_height,
                p2=self.patch_width,
            )
        else:
            y = rearrange(
                y,
                " b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.num_h_patch,
                w=self.num_w_patch,
                p1=self.patch_height,
                p2=self.patch_width,
            )

        if self.dct_block_size > 0:
            y = block_idct_2d(y, norm="ortho", block_size=self.dct_block_size)

        return y
