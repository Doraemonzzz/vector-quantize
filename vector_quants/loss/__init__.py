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

from .utils import get_post_transform

perceptual_loss_type_dict = {2: "alex", 3: "squeeze", 4: "vgg"}


def get_perceptual_loss(perceptual_loss_type):
    if perceptual_loss_type == 0:
        logging_info(f"Perceptual loss: None")
        model = None
    else:
        if perceptual_loss_type == 1:
            logging_info(f"Perceptual loss: Vgg lpips from fsq_pytorch")
            from .lpips import LPIPS

            model = LPIPS().eval()
        else:
            net_type = perceptual_loss_type_dict[perceptual_loss_type]
            logging_info(f"Perceptual loss: {net_type} lpips")
            from .perceptual_loss import LPIPS

            model = LPIPS(net_type=net_type).eval()

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


class Loss(nn.Module):
    def __init__(
        self,
        perceptual_loss_type=0,
        adversarial_loss_type=0,
        # weight
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_loss_weight=1.0,
        adversarial_loss_weight=0.0,
        codebook_loss_weight=1.0,
        entropy_loss_weight=0.0,
        sample_entropy_loss_weight=0.0,
        codebook_entropy_loss_weight=0.0,
        kl_loss_weight=5e-4,
        wm_l1_loss_weight=1.0,
    ):
        super().__init__()
        self.perceptual_loss = get_perceptual_loss(perceptual_loss_type)
        self.adversarial_loss = get_adversarial_loss(adversarial_loss_type)

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.sample_entropy_loss_weight = sample_entropy_loss_weight
        self.codebook_entropy_loss_weight = codebook_entropy_loss_weight
        self.wm_l1_loss_weight = wm_l1_loss_weight

    @property
    def keys(self):
        train_keys = [
            "l1_loss",
            "l2_loss",
            "perceptual_loss",
            "adversarial_loss",
            "codebook_loss",
            "commitment_loss",
            "entropy_loss",
            "sample_entropy_loss",
            "codebook_entropy_loss",
            "kl_loss",
            "loss",
            "wm_l1_loss",
        ]
        valid_keys = ["valid_" + key for key in train_keys]
        keys = train_keys + valid_keys
        return keys

    def forward(self, images, reconstructions, **kwargs):
        l1_loss = self.compute_l1_loss(images, reconstructions)
        l2_loss = self.compute_l2_loss(images, reconstructions)
        perceptual_loss = self.compute_perceptual_loss(images, reconstructions)
        adversarial_loss = self.compute_adversarial_loss(images, reconstructions)
        codebook_loss = kwargs.get("codebook_loss", torch.tensor(0.0).cuda().float())
        commitment_loss = kwargs.get(
            "commitment_loss", torch.tensor(0.0).cuda().float()
        )
        entropy_loss = kwargs.get("entropy_loss", torch.tensor(0.0).cuda().float())
        kl_loss = kwargs.get("kl_loss", torch.tensor(0.0).cuda().float())
        sample_entropy_loss = kwargs.get(
            "sample_entropy_loss", torch.tensor(0.0).cuda().float()
        )
        codebook_entropy_loss = kwargs.get(
            "codebook_entropy_loss", torch.tensor(0.0).cuda().float()
        )

        wm_l1_loss = kwargs.get("wm_l1_loss", torch.tensor(0.0).cuda().float())

        loss = (
            self.l1_loss_weight * l1_loss
            + self.l2_loss_weight * l2_loss
            + self.perceptual_loss_weight * perceptual_loss
            + self.adversarial_loss_weight * adversarial_loss
            + self.codebook_loss_weight * codebook_loss
            + self.entropy_loss_weight * entropy_loss
            + self.kl_loss_weight * kl_loss
            + self.sample_entropy_loss_weight * sample_entropy_loss
            + self.codebook_entropy_loss_weight * codebook_entropy_loss
            + self.wm_l1_loss_weight * wm_l1_loss
        )

        loss_dict = {
            "l1_loss": l1_loss.cpu().item(),
            "l2_loss": l2_loss.cpu().item(),
            "perceptual_loss": perceptual_loss.cpu().item(),
            "adversarial_loss": adversarial_loss.cpu().item(),
            "codebook_loss": codebook_loss.cpu().item(),
            "commitment_loss": commitment_loss.cpu().item(),
            "entropy_loss": entropy_loss.cpu().item(),
            "sample_entropy_loss": sample_entropy_loss.cpu().item(),
            "codebook_entropy_loss": codebook_entropy_loss.cpu().item(),
            "kl_loss": kl_loss.cpu().item(),
            "loss": loss.cpu().item(),
            "wm_l1_loss": wm_l1_loss.cpu().item(),
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
