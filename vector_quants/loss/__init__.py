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

from .gan_loss import (
    AUTO_DISC_LOSS_MAPPING,
    AUTO_DISC_MAPPING,
    AUTO_GEN_LOSS_MAPPING,
    AUTO_GP_LOSS_MAPPING,
)
from .perceptual_loss import LPIPS, LpipsTimm
from .utils import get_post_transform

perceptual_loss_type_dict = {2: "alex", 3: "squeeze", 4: "vgg"}


def get_perceptual_loss(perceptual_loss_type, perceptual_model_name):
    if perceptual_loss_type == 0:
        logging_info(f"Perceptual loss: None")
        model = None
    else:
        if perceptual_loss_type == 1:
            logging_info(f"Perceptual loss: Vgg lpips from fsq_pytorch")
            from .lpips import LPIPS

            model = LPIPS().eval()
        elif perceptual_loss_type in [2, 3, 4]:
            net_type = perceptual_loss_type_dict[perceptual_loss_type]
            logging_info(f"Perceptual loss: {net_type} lpips")

            model = LPIPS(net_type=net_type).eval()
        else:
            model = LpipsTimm(model_name=perceptual_model_name).eval()

        model.cuda(torch.cuda.current_device())

    return model


class Loss(nn.Module):
    def __init__(
        self,
        perceptual_loss_type=0,
        perceptual_model_name="resnet18.a1_in1k",
        # weight
        l1_loss_weight=1.0,
        l2_loss_weight=0.0,
        perceptual_loss_weight=1.0,
        codebook_loss_weight=1.0,
        entropy_loss_weight=0.0,
        sample_entropy_loss_weight=0.0,
        codebook_entropy_loss_weight=0.0,
        kl_loss_weight=5e-4,
        wm_l1_loss_weight=1.0,
        # d loss
        disc_loss_start_iter=20000,
        disc_type=None,
        gen_loss_type="hinge",
        gen_loss_weight=0.1,
        disc_loss_type="hinge",
        disc_loss_weight=1.0,
        disc_model_name="vit_small_patch16_224_dino",
        gp_loss_type="none",
        gp_loss_weight=0,
        in_channels=3,
        image_size=128,
    ):
        super().__init__()
        self.perceptual_loss = get_perceptual_loss(
            perceptual_loss_type, perceptual_model_name
        )

        self.l1_loss_weight = l1_loss_weight
        self.l2_loss_weight = l2_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.sample_entropy_loss_weight = sample_entropy_loss_weight
        self.codebook_entropy_loss_weight = codebook_entropy_loss_weight
        self.wm_l1_loss_weight = wm_l1_loss_weight
        # d loss
        self.disc_loss_start_iter = disc_loss_start_iter
        self.disc_type = disc_type
        self.gen_loss = AUTO_GEN_LOSS_MAPPING[gen_loss_type]
        self.gen_loss_weight = gen_loss_weight
        self.disc_loss = AUTO_DISC_LOSS_MAPPING[disc_loss_type]
        self.disc_loss_weight = disc_loss_weight
        if self.disc_type != "none":
            self.discriminator = AUTO_DISC_MAPPING[self.disc_type](
                input_nc=in_channels, image_size=image_size, model_name=disc_model_name
            )
        else:
            self.discriminator = None

        self.gp_loss_type = gp_loss_type
        self.gp_loss_weight = gp_loss_weight
        if self.gp_loss_type != "none":
            assert self.disc_type != "dino", "dino does not support gradient penalty"
            self.gp_loss = AUTO_GP_LOSS_MAPPING[self.gp_loss_type]
        else:
            self.gp_loss = None

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
            # d loss
            "gen_loss",
            "disc_loss",
            "gp_loss",
            "logits_real",
            "logits_fake",
        ]
        valid_keys = ["valid_" + key for key in train_keys]
        ema_keys = ["ema_valid_" + key for key in train_keys]
        keys = train_keys + valid_keys + ema_keys
        return keys

    def train(self, mode=True):
        # This can make sure that the perceptual model is always in eval mode
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.perceptual_loss is not None:
            self.perceptual_loss.train(False)

        return self

    def eval(self):
        return self.train(False)

    def use_disc(self, num_iter):
        return num_iter >= self.disc_loss_start_iter and self.discriminator != None

    def forward(
        self, images, reconstructions, num_iter=0, is_disc=False, scale=1, **kwargs
    ):
        if is_disc:
            disc_loss = torch.tensor(0.0).cuda().float()
            gp_loss = torch.tensor(0.0).cuda().float()
            logits_real = torch.tensor(0.0).cuda().float()
            logits_fake = torch.tensor(0.0).cuda().float()
            if self.use_disc(num_iter) and self.training:  # avoid eval bug
                if self.gp_loss_weight > 0:
                    images.requires_grad_()

                logits_real = self.discriminator(images)
                logits_fake = self.discriminator(reconstructions.detach())
                disc_loss = self.disc_loss(logits_real, logits_fake)

                if (
                    self.gp_loss is not None
                    and self.gp_loss_weight > 0
                    and self.training
                ):
                    gp_loss = self.gp_loss(images, logits_real, scale)

            loss_dict = {
                "disc_loss": disc_loss.cpu().item(),
                "gp_loss": gp_loss.cpu().item(),
                "logits_fake": logits_fake.mean().cpu().item(),
                "logits_real": logits_real.mean().cpu().item(),
            }

            loss = self.disc_loss_weight * disc_loss + self.gp_loss_weight * gp_loss

            return loss, loss_dict
        else:
            l1_loss = self.compute_l1_loss(images, reconstructions)
            l2_loss = self.compute_l2_loss(images, reconstructions)
            perceptual_loss = self.compute_perceptual_loss(images, reconstructions)
            codebook_loss = kwargs.get(
                "codebook_loss", torch.tensor(0.0).cuda().float()
            )
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

            gen_loss = torch.tensor(0.0).cuda().float()
            if self.use_disc(num_iter) and self.training:  # avoid eval bug
                logits_fake = self.discriminator(reconstructions)
                gen_loss = self.gen_loss(logits_fake)

            loss = (
                self.l1_loss_weight * l1_loss
                + self.l2_loss_weight * l2_loss
                + self.perceptual_loss_weight * perceptual_loss
                + self.codebook_loss_weight * codebook_loss
                + self.entropy_loss_weight * entropy_loss
                + self.kl_loss_weight * kl_loss
                + self.sample_entropy_loss_weight * sample_entropy_loss
                + self.codebook_entropy_loss_weight * codebook_entropy_loss
                + self.wm_l1_loss_weight * wm_l1_loss
                + self.gen_loss_weight * gen_loss
            )

            loss_dict = {
                "l1_loss": l1_loss.cpu().item(),
                "l2_loss": l2_loss.cpu().item(),
                "perceptual_loss": perceptual_loss.cpu().item(),
                "codebook_loss": codebook_loss.cpu().item(),
                "commitment_loss": commitment_loss.cpu().item(),
                "entropy_loss": entropy_loss.cpu().item(),
                "sample_entropy_loss": sample_entropy_loss.cpu().item(),
                "codebook_entropy_loss": codebook_entropy_loss.cpu().item(),
                "kl_loss": kl_loss.cpu().item(),
                "loss": loss.cpu().item(),
                "wm_l1_loss": wm_l1_loss.cpu().item(),
                "gen_loss": gen_loss.cpu().item(),
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
