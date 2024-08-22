import os

import torch
from torchvision.utils import make_grid, save_image

from vector_quants.generate import sample
from vector_quants.models import AutoArModel, AutoVqVae

prefix_stage1 = (
    "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/checkpoints"
)
# name_stage1 = "Vq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-256-mn-freq_transformer-hc-192-md-512-nl-12-ps-8-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-rs-1-num_embed-1024-num_embed-1024"
# name_stage1 = "Cvq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-64-mn-freq_transformer-hc-192-md-512-nl-12-ps-4-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-b-8-n-4-num_embed-8"
name_stage1 = "Cvq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-64-mn-freq_transformer-hc-192-md-512-nl-12-ps-4-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-b-8-n-4-num_embed-8"
prefix_stage2 = (
    "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/script/checkpoints_ar"
)
# name_stage2 = "Vq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-256-mn-freq_transformer-hc-192-md-512-nl-12-ps-8-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-rs-1-num_embed-1024-num_embed-1024-use_group_id-False"
# name_stage2 = "Cvq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-64-mn-freq_transformer-hc-192-md-512-nl-12-ps-4-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-b-8-n-4-num_embed-8-use_group_id-False"
name_stage2 = "Cvq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-loss-1-ga-1-cm-1-ed-64-mn-freq_transformer-hc-192-md-512-nl-12-ps-4-net-freq-dbs-0-uz-False-ufq-False-lre-1-qs-True-b-8-n-4-num_embed-8-use_group_id-True"
save = "samples/"

ckpt_path_stage1 = f"{prefix_stage1}/{name_stage1}/ckpts/99.pt"
ckpt_path_stage2 = f"{prefix_stage2}/{name_stage2}/ckpts/49.pt"

# embed_dim_stage1 = 256
# sample_step = 256
# use_group_id = False

# embed_dim_stage1 = 64
# sample_step = 64
# use_group_id = False

embed_dim_stage1 = 64
sample_step = 64
use_group_id = True


device = torch.cuda.current_device()

vqvae, vqvae_config, res = AutoVqVae.from_pretrained(
    ckpt_path_stage1,
    embed_dim_stage1=embed_dim_stage1,
)
vqvae = vqvae.to(device)
print(vqvae)
model = AutoArModel.from_pretrained(ckpt_path_stage2)
model = model.to(device)
print(model)

class_idx = torch.tensor(list(range(16))).to(device)
idx = sample(model, class_idx, sample_step)
generate_img = vqvae.indice_to_img(idx, use_group_id=use_group_id)

save_dir = os.path.join(save, name_stage2)
os.makedirs(save_dir, exist_ok=True)
save_image(
    make_grid(
        torch.cat([generate_img[:16]]),
        nrow=8,
    ),
    os.path.join(save, f"{name_stage2}/samples.jpg"),
    normalize=True,
)
