import os
from dataclasses import asdict
from pprint import pformat

import torch
import torch.distributed as dist
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.logger import Logger
from vector_quants.loss import Loss, get_post_transform
from vector_quants.metrics import CodeBookMetric, Metrics, metrics_names
from vector_quants.models import AutoVqVae
from vector_quants.utils import (
    compute_num_patch,
    create_npz_from_sample_folder,
    get_metrics_list,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    reduce_dict,
    set_random_seed,
    type_dict,
    update_dict,
)

from .base_evaluator import BaseEvaluator


class VQEvaluator(BaseEvaluator):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        logging_info(pformat(asdict(cfg)))

        cfg_model_stage2 = cfg.model_stage2
        cfg_model = cfg.model
        cfg_train = cfg.train
        cfg_data = cfg.data
        cfg_loss = cfg.loss
        cfg_sample = cfg.sample
        # for transformer
        cfg_model.image_size = cfg_data.image_size
        cfg_model.num_patch = compute_num_patch(cfg_model)

        set_random_seed(cfg_train.seed)

        distributed.enable(overwrite=True)
        cfg_train.distributed = True
        cfg_train.gpu = int(os.environ["LOCAL_RANK"])
        cfg_train.world_size = int(os.environ["WORLD_SIZE"])

        mkdir_ckpt_dirs(cfg_train)

        # 1, load dataset
        self.train_data_loader = get_data_loaders(cfg_train, cfg_data, is_train=True)
        self.val_data_loader = get_data_loaders(cfg_train, cfg_data, is_train=False)

        # 2, load model
        self.dtype = type_dict[cfg_train.dtype]
        vqvae, vqvae_config, res = AutoVqVae.from_pretrained(
            cfg_train.ckpt_path_stage1,
            embed_dim_stage1=cfg_model_stage2.embed_dim_stage1,
        )
        self.model = vqvae
        self.model.cuda(torch.cuda.current_device())
        self.model.eval()

        logging_info(res)
        logging_info(self.model)

        # 3. get loss
        self.loss_fn = Loss(
            perceptual_loss_type=cfg_loss.perceptual_loss_type,
            perceptual_model_name=cfg_loss.perceptual_model_name,
            l1_loss_weight=cfg_loss.l1_loss_weight,
            l2_loss_weight=cfg_loss.l2_loss_weight,
            perceptual_loss_weight=cfg_loss.perceptual_loss_weight,
            codebook_loss_weight=cfg_loss.codebook_loss_weight,
            entropy_loss_weight=cfg_model.entropy_loss_weight,
            kl_loss_weight=cfg_model.kl_loss_weight,
            sample_entropy_loss_weight=cfg_model.sample_entropy_loss_weight,
            codebook_entropy_loss_weight=cfg_model.codebook_entropy_loss_weight,
            wm_l1_loss_weight=cfg_loss.wm_l1_loss_weight,
            # d loss
            disc_loss_start_iter=cfg_loss.disc_loss_start_iter,
            disc_type=cfg_loss.disc_type,
            gen_loss_type=cfg_loss.gen_loss_type,
            gen_loss_weight=cfg_loss.gen_loss_weight,
            disc_loss_type=cfg_loss.disc_loss_type,
            disc_loss_weight=cfg_loss.disc_loss_weight,
            disc_model_name=cfg_loss.disc_model_name,
            gp_loss_type=cfg_loss.gp_loss_type,
            gp_loss_weight=cfg_loss.gp_loss_weight,
            in_channels=cfg_model.in_channels,
            image_size=cfg_data.image_size,
        ).to(torch.cuda.current_device())
        logging_info(self.loss_fn)
        loss_fn_keys = self.loss_fn.keys

        # 4. ddp setup
        num_embed = self.model.num_embed

        # evaluation
        self.eval_metrics = Metrics(
            metrics_list=get_metrics_list(cfg_loss.metrics_list),
            dataset_name=cfg_data.data_set,
            device=torch.cuda.current_device(),
        )

        self.num_embed = num_embed
        self.codebook_metric = CodeBookMetric(self.num_embed)

        # logger
        self.logger = Logger(
            keys=["epoch", "iter", "lr", "d_lr"]
            + loss_fn_keys
            + metrics_names
            + ["gnorm", "grad_disc_norm"],
            use_wandb=cfg_train.use_wandb,
            cfg=cfg,
            wandb_entity=cfg_train.wandb_entity,
            wandb_project=cfg_train.wandb_project,
            wandb_exp_name=cfg_train.wandb_exp_name,
            wandb_cache_dir=cfg_train.wandb_cache_dir,
        )

        # other params
        self.max_train_epochs = cfg_train.max_train_epochs
        self.log_interval = cfg_train.log_interval
        self.eval_interval = cfg_train.eval_interval
        self.gradient_accumulation_steps = cfg_train.gradient_accumulation_steps
        self.save_interval = cfg_train.save_interval
        self.save = cfg_train.save
        self.is_llamagen = "llamagen" in cfg_train.ckpt_path_stage1
        self.is_any_diffusion = "any_diffusion" in cfg_train.ckpt_path_stage1
        if self.is_llamagen or self.is_any_diffusion:
            cfg_loss.post_transform_type = -1
            logging_info("Use llamagen post_transform")
        self.post_transform = get_post_transform(
            cfg_loss.post_transform_type, data_set=cfg_data.data_set
        )
        self.clip_grad = cfg_train.clip_grad
        self.num_sample = cfg_data.num_sample
        self.sample_folder_dir = cfg_data.sample_folder_dir
        self.val_folder_dir = cfg_data.val_folder_dir
        self.sample_step = cfg_sample.sample_step

    @property
    def is_main_process(self):
        return is_main_process()

    def eval(self):
        logging_info("Start Evaluation")
        self.model.eval()
        self.loss_fn.eval()
        self.eval_metrics.reset()

        loss_dict_total = {}
        cnt = 0
        world_size = dist.get_world_size()
        for input_img, _ in tqdm(
            self.val_data_loader, disable=not self.is_main_process
        ):
            cnt += input_img.shape[0] * world_size
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    input_img = input_img.cuda(torch.cuda.current_device())
                    if self.is_llamagen:
                        reconstructions, _ = self.model(input_img)
                        loss_dict = {}
                    else:
                        reconstructions, indices, loss_dict = self.model(input_img)
                    # rescale to [0, 1]
                    input_img = self.post_transform(input_img)
                    reconstructions = self.post_transform(reconstructions)
                    loss, loss_dict = self.loss_fn(
                        input_img,
                        reconstructions,
                        **loss_dict,
                    )

                loss_dict_total = update_dict(loss_dict_total, loss_dict)

            self.eval_metrics.update(
                real=input_img.contiguous(), fake=reconstructions.contiguous()
            )

            if not self.is_llamagen:
                try:
                    self.codebook_metric.update(indices)
                except:
                    pass

            if cnt >= self.num_sample:
                logging_info(f"End Evaluation, sample {cnt} images")
                break

        torch.cuda.synchronize()

        eval_loss_dict = reduce_dict(
            loss_dict_total, n=self.num_sample, prefix="valid_"
        )
        eval_results = self.eval_metrics.compute_and_reduce()
        if self.is_llamagen:
            codebook_results = {}
        else:
            try:
                codebook_results = self.codebook_metric.get_result()
            except:
                codebook_results = {}

        self.logger.log(**(eval_loss_dict | eval_results | codebook_results))

        torch.cuda.empty_cache()
        logging_info(f"End Evaluation, sample {cnt} images")

    def sample(self):
        logging_info("Start Sampling")
        self.model.eval()
        os.makedirs(self.sample_folder_dir, exist_ok=True)
        os.makedirs(self.val_folder_dir, exist_ok=True)

        cnt = 0
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        for input_img, _ in tqdm(
            self.val_data_loader, disable=not self.is_main_process
        ):
            global_batch_size = input_img.shape[0] * world_size
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    input_img = input_img.cuda(torch.cuda.current_device())
                    if self.is_llamagen:
                        reconstructions, _ = self.model(input_img)
                        loss_dict = {}
                    else:
                        reconstructions, indices, loss_dict = self.model(input_img)
                    reconstructions = self.post_transform(reconstructions)
                    input_img = self.post_transform(input_img)

                for i in range(input_img.shape[0]):
                    reconstruction = reconstructions[i]
                    reconstruction = rearrange(reconstruction, "c h w -> h w c")
                    index = i * world_size + rank + cnt
                    data = (
                        torch.clamp(255 * reconstruction, 0, 255)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )
                    Image.fromarray(data).save(
                        f"{self.sample_folder_dir}/{index:06d}.png"
                    )
                    if not os.path.exists(f"{self.val_folder_dir}/data.npz"):
                        input_img_ = input_img[i]
                        input_img_ = rearrange(input_img_, "c h w -> h w c")
                        input_img_ = (
                            torch.clamp(255 * input_img_, 0, 255)
                            .to("cpu", dtype=torch.uint8)
                            .numpy()
                        )
                        Image.fromarray(input_img_).save(
                            f"{self.val_folder_dir}/{index:06d}.png"
                        )

            cnt += global_batch_size
            if cnt >= self.num_sample:
                logging_info(f"End Sampling, sample {cnt} images")
                break

        # Make sure all processes have finished saving their samples before attempting to convert to .npz
        dist.barrier()
        if rank == 0:
            # create_npz_from_sample_folder(self.sample_folder_dir, self.num_sample)
            if not os.path.exists(f"{self.val_folder_dir}/data.npz"):
                create_npz_from_sample_folder(self.val_folder_dir, self.num_sample)
            logging_info("Done.")

        dist.barrier()
        dist.destroy_process_group()

    def sample_partial(self):
        logging_info("Start Sampling")
        self.model.eval()
        os.makedirs(self.sample_folder_dir, exist_ok=True)

        world_size = dist.get_world_size()
        dist.get_rank()

        for i, (input_img, _) in tqdm(
            enumerate(self.val_data_loader), disable=not self.is_main_process
        ):
            input_img.shape[0] * world_size
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    input_img = input_img.cuda(torch.cuda.current_device())
                    input_img_transform = self.post_transform(input_img)
                    step = self.sample_step
                    while step > 0:
                        if self.is_llamagen:
                            reconstructions, _ = self.model(input_img)
                            loss_dict = {}
                        else:
                            reconstructions, indices, loss_dict = self.model(
                                input_img, step=step
                            )
                        reconstructions = self.post_transform(reconstructions)

                        # save image for checking training
                        dist.barrier()
                        if self.is_main_process:
                            save_image(
                                make_grid(
                                    torch.cat([input_img_transform, reconstructions]),
                                    nrow=input_img.shape[0],
                                ),
                                os.path.join(self.sample_folder_dir, f"{i}_{step}.jpg"),
                            )
                        step //= 2

                        dist.barrier()

            break
