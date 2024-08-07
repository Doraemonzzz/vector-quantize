from .softmax_attention import SoftmaxAttention
from .softmax_attention_ar import SoftmaxAttentionAr

AUTO_TOKEN_MIXER_MAPPING = {
    "softmax": SoftmaxAttention,
    "softmax_ar": SoftmaxAttentionAr,
}
