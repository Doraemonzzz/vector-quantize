from .baseline import FSQ, LFQ, SFSQ, RQBottleneck, VectorQuantizeEMA
from .carry_vector_quantizer import CarryVectorQuantizer
from .ema_vector_quantizer import EMAVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer
from .group_vector_quantizer import GroupVectorQuantizer
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .hierachical_vector_quantizer import HierachicalVectorQuantizer
from .lookup_free_quantizer import LookUpFreeQuantizer
from .radial_quantizer import RadialQuantizer
from .residual_vector_quantizer import ResidualVectorQuantizer
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
}


def get_quantizer(args):
    if args.quantizer == "ema" or args.quantizer == "origin":
        quantizer = VectorQuantizeEMA(args, args.embed_dim, args.num_embed)

    elif args.quantizer == "lfq":
        quantizer = LFQ(
            codebook_size=2**args.lfq_dim,
            dim=args.lfq_dim,
            entropy_loss_weight=args.entropy_loss_weight,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "fsq":
        quantizer = FSQ(levels=args.levels)
    elif args.quantizer == "sfsq":
        quantizer = SFSQ(levels=args.levels)
    elif args.quantizer == "rvq":
        quantizer = RQBottleneck(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            num_residual=args.num_residual,
            decay=args.ema_decay,
            shared_codebook=args.shared_codebook,
        )
    elif args.quantizer in QUANTIZER_DICT:
        quantizer = QUANTIZER_DICT[args.quantizer](
            args,
        )

    return quantizer
