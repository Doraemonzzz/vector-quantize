from .carry_vector_quantizer import CarryVectorQuantizer
from .ema_vector_quantizer import EMAVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer
from .group_vector_quantizer import GroupVectorQuantizer
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .hierachical_vector_quantizer import HierachicalVectorQuantizer
from .identity_vector_quantizer import IdentityVectorQuantizer
from .lookup_free_quantizer import LookUpFreeQuantizer
from .radial_quantizer import RadialQuantizer
from .residual_finite_scalar_quantizer import ResidualFiniteScalarQuantizer
from .residual_vector_quantizer import ResidualVectorQuantizer
from .test import ResidualCoefQuantizer
from .vector_quantizer import VectorQuantizer

QUANTIZER_DICT = {
    "Vq": VectorQuantizer,
    "EmaVq": EMAVectorQuantizer,
    "GumbelVq": GumbelVectorQuantizer,
    "Gvq": GroupVectorQuantizer,
    "Hvq": HierachicalVectorQuantizer,
    "Cvq": CarryVectorQuantizer,
    "Rvq": ResidualVectorQuantizer,
    "Lfq": LookUpFreeQuantizer,
    "Fsq": FiniteScalarQuantizer,
    "Raq": RadialQuantizer,
    "Rfsq": ResidualFiniteScalarQuantizer,
    "Rcq": ResidualCoefQuantizer,
    "Ivq": IdentityVectorQuantizer,
}


def get_quantizer(args):
    quantizer = QUANTIZER_DICT[args.quantizer](
        args,
    )

    return quantizer
