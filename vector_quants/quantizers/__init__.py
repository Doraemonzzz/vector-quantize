from .carry_vector_quantizer import CarryVectorQuantizer
from .ema_vector_quantizer import EMAVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer
from .group_vector_quantizer import GroupVectorQuantizer
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .hierachical_vector_quantizer import HierachicalVectorQuantizer
from .lookup_free_quantizer import LookUpFreeQuantizer
from .radial_quantizer import RadialQuantizer
from .residual_finite_scalar_quantizer import ResidualFiniteScalarQuantizer
from .residual_vector_quantizer import ResidualVectorQuantizer
from .softmax_vector_quantizer import SoftmaxVectorQuantizer
from .vector_quantizer import VectorQuantizer

QUANTIZER_DICT = {
    "Vq": VectorQuantizer,
    "EmaVq": EMAVectorQuantizer,
    "GumbelVq": GumbelVectorQuantizer,
    "SoftmaxVq": SoftmaxVectorQuantizer,
    "Gvq": GroupVectorQuantizer,
    "Hvq": HierachicalVectorQuantizer,
    "Cvq": CarryVectorQuantizer,
    "Rvq": ResidualVectorQuantizer,
    "Lfq": LookUpFreeQuantizer,
    "Fsq": FiniteScalarQuantizer,
    "Raq": RadialQuantizer,
    "Rfsq": ResidualFiniteScalarQuantizer,
}


def get_quantizer(args):
    quantizer = QUANTIZER_DICT[args.quantizer](
        args,
    )

    return quantizer
