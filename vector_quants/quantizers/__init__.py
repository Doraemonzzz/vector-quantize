from .baseline import FSQ, LFQ, SFSQ, RQBottleneck, VectorQuantizeEMA
from .carry_vector_quantizer import CarryVectorQuantizer
from .ema_vector_quantizer import EMAVectorQuantizer
from .finite_scalar_quantizer import FiniteScalarQuantizer
from .group_vector_quantizer import GroupVectorQuantizer
from .gumbel_vector_quantizer import GumbelVectorQuantizer
from .hierachical_vector_quantizer import HierachicalVectorQuantizer
from .lookup_free_quantizer import LookUpFreeQuantizer
from .residual_vector_quantizer import ResidualVectorQuantizer
from .vector_quantizer import VectorQuantizer


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
    elif args.quantizer == "Vq":
        quantizer = VectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "EmaVq":
        quantizer = EMAVectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
            decay=args.ema_decay,
        )
    elif args.quantizer == "GumbelVq":
        quantizer = GumbelVectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
            temp=args.temp,
            kl_loss_weight=args.kl_loss_weight,
        )
    elif args.quantizer == "Gvq":
        quantizer = GroupVectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            num_group=args.num_group,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "Hvq":
        quantizer = HierachicalVectorQuantizer(
            levels=args.levels,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "Cvq":
        quantizer = CarryVectorQuantizer(
            base=args.base,
            num_levels=args.num_levels,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "Rvq":
        quantizer = ResidualVectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            num_residual=args.num_residual,
            commitment_loss_weight=args.commitment_loss_weight,
        )
    elif args.quantizer == "Lfq":
        quantizer = LookUpFreeQuantizer(
            embed_dim=args.embed_dim,
        )
    elif args.quantizer == "Fsq":
        quantizer = FiniteScalarQuantizer(
            levels=args.levels,
        )

    return quantizer
