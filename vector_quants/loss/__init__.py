import torch
import torch.nn as nn

from vector_quants.data import get_mean_std_from_dataset_name
from vector_quants.utils import (
    get_args,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    rescale_image_tensor,
    set_random_seed,
    type_dict,
)

from .lpips import LPIPS
from .metric import get_revd_perceptual, transform_rev


def get_perceptual_loss(perceptual_loss_type):
    if perceptual_loss_type == 0:
        logging_info(f"Perceptual loss: None")
        model = None
    else:
        if perceptual_loss_type == 1:
            logging_info(f"Perceptual loss: Vgg lpips from fsq_pytorch")
            model = LPIPS().eval()

        model.cuda(torch.cuda.current_device())

    return model


def get_adversarial_loss(adversarial_loss_type):
    if adversarial_loss_type == 0:
        logging_info(f"Adversarial loss: None")
        model = None
    else:
        logging_info(f"Adversarial loss: None")
        model = None

    return model


def get_post_transform(post_transform_type, data_set="imagenet-1k"):
    if post_transform_type == 0:
        logging_info(f"Post Transform: None")
        post_transform = lambda x: x
    elif post_transform_type == 1:
        logging_info(f"Post Transform: fsq_pytorch default")
        post_transform = transform_rev
    else:
        logging_info(f"Post Transform: rescale to [0, 1]")
        norm_mean, norm_std = get_mean_std_from_dataset_name(data_set)
        post_transform = lambda x: rescale_image_tensor(x, norm_mean, norm_std)

    return post_transform


class Loss(nn.Module):
    def __init__(
        self,
        perceptual_loss_type=0,
        adversarial_loss_type=0,
        # weight
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_loss_weight=0.0,
        adversarial_loss_weight=0.0,
        codebook_loss_weight=1.0,
    ):
        super().__init__()
        self.perceptual_loss = get_perceptual_loss(perceptual_loss_type)
        self.adversarial_loss = get_adversarial_loss(adversarial_loss_type)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.codebook_loss_weight = codebook_loss_weight

    def forward(self, codebook_loss, images, reconstructions):
        l1_loss = self.compute_l1_loss(images, reconstructions)
        l2_loss = self.compute_l2_loss(images, reconstructions)
        perceptual_loss = self.compute_perceptual_loss(images, reconstructions)
        adversarial_loss = self.compute_adversarial_loss(images, reconstructions)

        loss = (
            self.l1_loss_weight * l1_loss
            + self.l2_loss_weight * l2_loss
            + self.perceptual_loss_weight * perceptual_loss
            + self.adversarial_loss_weight * adversarial_loss
            + self.codebook_loss_weight * codebook_loss
        )

        loss_dict = {
            "l1_loss": l1_loss.cpu().item(),
            "l2_loss": l2_loss.cpu().item(),
            "perceptual_loss": perceptual_loss.cpu().item(),
            "adversarial_loss": adversarial_loss.cpu().item(),
            "codebook_loss": codebook_loss.cpu().item(),
            "loss": loss.cpu().item(),
        }

        return loss, loss_dict

    def compute_l1_loss(self, images, reconstructions):
        if self.l1_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        else:
            loss = (reconstructions - images).abs().mean()

        return loss

    def compute_l2_loss(self, images, reconstructions):
        if self.l2_loss_weight == 0:
            loss = torch.tensor(0.0).cuda().float()
        else:
            loss = (reconstructions - images).pow(2).mean()

        return loss

    def compute_perceptual_loss(self, images, reconstructions):
        if self.perceptual_loss_weight == 0 or self.perceptual_loss is None:
            loss = torch.tensor(0.0).cuda().float()
        else:
            loss = self.perceptual_loss(images, reconstructions)

        return loss.mean()

    def compute_adversarial_loss(self, images, reconstructions):
        if self.adversarial_loss_weight == 0 or self.adversarial_loss is None:
            loss = torch.tensor(0.0).cuda().float()
        else:
            loss = self.perceptual_loss(images, reconstructions)

        return loss.mean()
