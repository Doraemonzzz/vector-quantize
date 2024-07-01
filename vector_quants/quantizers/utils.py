import torch
import torch.nn.functional as F
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


def entropy_loss_fn(affinity, temperature, loss_type="softmax", eps=1e-5):
    """
    Increase codebook usage by maximizing entropy

    affinity: 2D tensor of size Dim, n_classes
    """

    n_classes = affinity.shape[-1]

    affinity = torch.div(affinity, temperature)
    probs = F.softmax(affinity, dim=-1)
    log_probs = F.log_softmax(affinity + eps, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(affinity, dim=-1)
        one_hots = F.one_hot(codes, n_classes).to(codes)
        one_hots = probs - (probs - one_hots).detach()
        target_probs = one_hots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))

    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    return sample_entropy - avg_entropy


# def entropy_loss_fn(affinity, temperature, loss_type="softmax", eps=1e-5):
#     """
#     Increase codebook usage by maximizing entropy

#     affinity: 2D tensor of size Dim, n_classes
#     """

#     n_classes = affinity.shape[-1]

#     affinity = torch.div(affinity, temperature)
#     log_probs = F.log_softmax(affinity + eps, dim=-1)
#     probs = torch.exp(log_probs)

#     if loss_type == "softmax":
#         target_probs = probs
#     elif loss_type == "argmax":
#         codes = torch.argmax(affinity, dim=-1)
#         one_hots = F.one_hot(codes, n_classes).to(codes)
#         one_hots = probs - (probs - one_hots).detach()
#         target_probs = one_hots
#     else:
#         raise ValueError("Entropy loss {} not supported".format(loss_type))

#     avg_probs = torch.mean(target_probs, dim=0)
#     avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
#     sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
#     return sample_entropy - avg_entropy


def kl_loss_fn(affinity, eps=1e-5):
    # kl(p || uniform) = -entropy(p) + c
    log_probs = F.log_softmax(affinity + eps, dim=-1)
    probs = torch.exp(log_probs)
    kl_loss = torch.mean(torch.sum(probs * log_probs, dim=-1))

    return kl_loss
