import torch
from einops import pack, unpack


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def compute_dist(x, y):
    # x: n, d
    # y: m, d
    y_t = y.t()

    # |x - y| ^ 2 = x * x ^ t + y * y ^ t - 2 * x * y ^ t
    dist = (
        torch.sum(x**2, dim=-1, keepdim=True)
        + torch.sum(y_t**2, dim=0, keepdim=True)
        - 2 * torch.matmul(x, y_t)
    )

    return dist


def round_ste(x):
    """Round with straight through gradients."""
    xhat = x.round()
    return x + (xhat - x).detach()
