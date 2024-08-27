from .block_dct_patch_embed import BlockDctPatchEmbed, ReverseBlockDctPatchEmbed
from .freq_patch_embed import FreqPatchEmbed, ReverseFreqPatchEmbed
from .patch_embed import PatchEmbed, ReversePatchEmbed

AUTO_PATCH_EMBED_MAPPING = {
    "vanilla": PatchEmbed,
    "freq": FreqPatchEmbed,
    "block_dct": BlockDctPatchEmbed,
}

AUTO_REVERSE_PATCH_EMBED_MAPPING = {
    "vanilla": ReversePatchEmbed,
    "freq": ReverseFreqPatchEmbed,
    "block_dct": ReverseBlockDctPatchEmbed,
}
