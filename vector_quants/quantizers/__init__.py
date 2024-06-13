from .baseline import FSQ, LFQ, SFSQ, VectorQuantizeEMA
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
    elif args.quantizer == "Vq":
        quantizer = VectorQuantizer(
            num_embed=args.num_embed,
            embed_dim=args.embed_dim,
            commitment_loss_weight=args.commitment_loss_weight,
        )

    return quantizer
