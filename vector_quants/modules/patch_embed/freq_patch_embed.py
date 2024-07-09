from einops import rearrange
from torch import nn

from vector_quants.ops import zigzag_indices
from vector_quants.utils import pair


class FreqPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch

        self.to_patch_embedding = nn.Linear(
            channels * patch_height * patch_width, embed_dim, bias=bias
        )
        # indices, reverse_indices = zigzag_indices(self.num_h_patch, self.num_w_patch)
        indices, reverse_indices = zigzag_indices(image_height, image_width)
        self.register_buffer("indices", indices, persistent=False)

    # def forward(self, x):
    #     y = rearrange(
    #         x,
    #         "b c (p1 h) (p2 w) -> b (p1 p2 c) h w",
    #         h=self.num_h_patch,
    #         w=self.num_w_patch,
    #     )
    #     y = dct_2d(y, norm="ortho")
    #     y = rearrange(y, "b d h w -> b (h w) d")[:, self.indices]
    #     y = self.to_patch_embedding(y)

    #     return y

    def forward(self, x):
        y = rearrange(
            x,
            "b c h w -> b c (h w)",
        )[:, :, self.indices]

        y = rearrange(y, "b c (n d) -> b n (d c)", n=self.num_patch)
        y = self.to_patch_embedding(y)

        return y


class ReverseFreqPatchEmbed(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        embed_dim,
        channels=3,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_h_patch = image_height // patch_height
        self.num_w_patch = image_width // patch_width
        self.num_patch = self.num_h_patch * self.num_w_patch
        self.image_height = image_height

        self.reverse_patch_embedding = nn.Linear(
            embed_dim, channels * patch_height * patch_width, bias=bias
        )
        # indices, reverse_indices = zigzag_indices(self.num_h_patch, self.num_w_patch)
        indices, reverse_indices = zigzag_indices(image_height, image_width)
        self.register_buffer("reverse_indices", reverse_indices, persistent=False)

    # def forward(self, x):
    #     y = self.reverse_patch_embedding(x)[:, self.reverse_indices]
    #     y = rearrange(
    #         y,
    #         "b (h w) d -> b d h w",
    #         h=self.num_h_patch,
    #     )
    #     y = idct_2d(y, norm="ortho")

    #     y = rearrange(
    #         y,
    #         "b (p1 p2 c) h w -> b c (p1 h) (p2 w)",
    #         p1=self.patch_height,
    #         p2=self.patch_width,
    #     )

    #     return y

    def forward(self, x):
        y = self.reverse_patch_embedding(x)
        y = rearrange(
            y, "b n (d c) -> b c (n d)", d=self.patch_height * self.patch_width
        )[:, :, self.reverse_indices]
        y = rearrange(y, "b c (h w) -> b c h w", h=self.image_height)

        return y
