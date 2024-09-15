# credit to: https://github.com/FoundationVision/LlamaGen/blob/main/tokenizer/tokenizer_image/vq_loss.py
import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logits_real), logits_real)
    )
    loss_fake = torch.mean(
        F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake)
    )
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake), logit_fake)
    )
