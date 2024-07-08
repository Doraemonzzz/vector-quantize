from .freq_patch_embed import FreqPatchEmbed, ReverseFreqPatchEmbed
from .patch_embed import PatchEmbed, ReversePatchEmbed

AUTO_PATCH_EMBED_MAPPING = {
    "vanilla": PatchEmbed,
    "freq": FreqPatchEmbed,
}

AUTO_REVERSE_PATCH_EMBED_MAPPING = {
    "vanilla": ReversePatchEmbed,
    "freq": ReverseFreqPatchEmbed,
}
