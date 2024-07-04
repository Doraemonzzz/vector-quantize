from .ffn import FFN
from .glu import GLU

AUTO_CHANNEL_MIXER_MAPPING = {"glu": GLU, "ffn": FFN}
