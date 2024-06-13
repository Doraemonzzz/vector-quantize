from .baseline import VQVAE

MODEL_DICT = {"baseline": VQVAE}


def get_model(args):
    return MODEL_DICT[args.model_name](args)
