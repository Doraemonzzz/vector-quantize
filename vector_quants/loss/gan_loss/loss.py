# credit to: https://github.com/FoundationVision/LlamaGen/blob/main/tokenizer/tokenizer_image/vq_loss.py
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import grad as torch_grad


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


def hinge_gen_loss(logits_fake):
    return -torch.mean(logits_fake)


def vanilla_gen_loss(logits_fake):
    return torch.mean(F.softplus(-logits_fake))


def non_saturating_gen_loss(logits_fake):
    return torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logits_fake), logits_fake)
    )


def gradient_penalty_l2_loss(images, output, scale=1):
    gradients = (
        torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        / scale
    )

    gradients = rearrange(gradients, "b ... -> b (...)")
    return (gradients.norm(2, dim=1) ** 2).mean()


def gradient_penalty_wgan_loss(images, output, scale=1):
    gradients = (
        torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        / scale
    )

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
