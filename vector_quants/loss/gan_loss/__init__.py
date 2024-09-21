from .discriminator_patchgan import PatchGANDiscriminator
from .discriminator_stylegan import StyleGANDiscriminator
from .loss import (
    gradient_penalty_l2_loss,
    gradient_penalty_wgan_loss,
    hinge_d_loss,
    hinge_gen_loss,
    non_saturating_d_loss,
    non_saturating_gen_loss,
    vanilla_d_loss,
)

AUTO_DISC_MAPPING = {
    "patchgan": PatchGANDiscriminator,
    "stylegan": StyleGANDiscriminator,
}

AUTO_GEN_LOSS_MAPPING = {
    "hinge": hinge_gen_loss,
    "non-saturating": non_saturating_gen_loss,
}

AUTO_DISC_LOSS_MAPPING = {
    "hinge": hinge_d_loss,
    "vanilla": vanilla_d_loss,
    "non-saturating": non_saturating_d_loss,
}

AUTO_GP_LOSS_MAPPING = {
    "wgan": gradient_penalty_wgan_loss,
    "l2": gradient_penalty_l2_loss,
}
