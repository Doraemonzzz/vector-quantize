import os
from dataclasses import asdict
from pprint import pformat

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.logger import Logger
from vector_quants.loss import get_post_transform
from vector_quants.metrics import Metrics, metrics_names
from vector_quants.models import AutoArModel, AutoVqVae
from vector_quants.scheduler import CfgScheduler
from vector_quants.utils import (
    compute_num_patch,
    get_metrics_list,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    set_random_seed,
    type_dict,
    update_dict,
)

from .base_evaluator import BaseEvaluator


class AREvaluator(BaseEvaluator):
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
        cfg_model.sample_step = cfg_sample.sample_step

        set_random_seed(cfg_train.seed)

        distributed.enable(overwrite=True)
        cfg_train.distributed = True
        cfg_train.gpu = int(os.environ["LOCAL_RANK"])
        cfg_train.world_size = int(os.environ["WORLD_SIZE"])

        mkdir_ckpt_dirs(cfg_train)

        # 1, load dataset
        self.indice_loader = get_data_loaders(
            cfg_train, cfg_data, is_train=False, is_indice=True
        )
        # 2, load model
        # vqvae
        self.dtype = type_dict[cfg_train.dtype]
        vqvae, vqvae_config, res = AutoVqVae.from_pretrained(
            cfg_train.ckpt_path_stage1,
            embed_dim_stage1=cfg_model_stage2.embed_dim_stage1,
        )
        self.vqvae = vqvae
        self.vqvae.cuda(torch.cuda.current_device())
        self.vqvae.eval()
        logging_info(res)
        logging_info(self.vqvae)
        # ar
        self.model = AutoArModel.from_pretrained(
            cfg_train.ckpt_path_stage2,
        )
        self.model.cuda(torch.cuda.current_device())
        logging_info(self.model)

        # 3. ddp setup
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[cfg_train.gpu],
        )

        # evaluation
        metrics_list = get_metrics_list(cfg_loss.metrics_list)
        assert (
            len(metrics_list) == 1 and metrics_list[0] == "fid"
        ), "Only fid-50k is supported for now"
        self.eval_metrics = Metrics(
            metrics_list=get_metrics_list(cfg_loss.metrics_list),
            dataset_name=cfg_data.data_set,
            device=torch.cuda.current_device(),
            reset_real_features=False,
            fid_statistics_file=cfg_train.ref_batch,
        )

        # logger
        self.logger = Logger(
            keys=["epoch", "iter", "lr", "d_lr"]
            + metrics_names
            + ["gnorm", "grad_disc_norm"],
            use_wandb=cfg_train.use_wandb,
            cfg=cfg,
            wandb_entity=cfg_train.wandb_entity,
            wandb_project=cfg_train.wandb_project,
            wandb_exp_name=cfg_train.wandb_exp_name,
            wandb_cache_dir=cfg_train.wandb_cache_dir,
        )
        # cfg scheduler
        self.cfg_scheduler = CfgScheduler(
            num_steps=cfg_sample.sample_step,
        )

        # other params
        self.max_train_epochs = cfg_train.max_train_epochs
        self.log_interval = cfg_train.log_interval
        self.eval_interval = cfg_train.eval_interval
        self.gradient_accumulation_steps = cfg_train.gradient_accumulation_steps
        self.save_interval = cfg_train.save_interval
        self.save = cfg_train.save
        self.is_llamagen = "llamagen" in cfg_train.ckpt_path_stage1
        if self.is_llamagen:
            cfg_loss.post_transform_type = -1
            logging_info("Use llamagen post_transform")
        self.post_transform = get_post_transform(
            cfg_loss.post_transform_type, data_set=cfg_data.data_set
        )
        self.num_sample = cfg_data.num_sample
        self.sample_folder_dir = cfg_data.sample_folder_dir
        self.model_type = cfg_model_stage2.model_name
        self.cfg_scale_list = cfg_sample.cfg_scale_list
        self.cfg_schedule_list = cfg_sample.cfg_schedule_list
        self.sample_step = cfg_sample.sample_step
        self.save_npz = cfg_sample.save_npz
        self.ref_batch = cfg_train.ref_batch

    @property
    def is_main_process(self):
        return is_main_process()

    def eval(self):
        self.model.eval()
        self.vqvae.eval()
        self.eval_metrics.reset()

        npy_dir = os.path.join(self.save, "npy_proc")
        eval_results_total = {}
        for cfg_schedule in self.cfg_schedule_list:
            for cfg_scale in self.cfg_scale_list:
                name = f"fid_{cfg_schedule}_{cfg_scale}"
                npy_proc = os.path.join(npy_dir, name)
                os.makedirs(npy_proc, exist_ok=True)
                save_img = None
                self.eval_metrics.reset()
                # set cfg_scale and cfg_schedule
                self.cfg_scheduler.reset(cfg_scale=cfg_scale, cfg_schedule=cfg_schedule)
                logging_info(
                    f"Eval cfg_scale: {cfg_scale}, cfg_schedule: {cfg_schedule}"
                )

                for i, class_idx in tqdm(
                    enumerate(self.indice_loader), disable=not self.is_main_process
                ):
                    class_idx = class_idx.cuda(torch.cuda.current_device())

                    with torch.no_grad():
                        with torch.amp.autocast(
                            device_type="cuda", dtype=self.dtype, enabled=False
                        ):
                            # only for test
                            # test_sample_with_kv_cache(self.model, class_idx, self.sample_step)
                            if self.model_type in ["transformer", "sg_transformer"]:
                                idx = self.model.module.generate(
                                    steps=self.sample_step,
                                    c=class_idx,
                                    cfg_scheduler=self.cfg_scheduler,
                                )
                            else:
                                idx = generate_llamagen(
                                    self.model,
                                    cond=class_idx,
                                    max_new_tokens=self.sample_step,
                                )
                            generate_img = self.vqvae.indice_to_img(idx)
                            # rescale to [0, 1]
                            generate_img_fid = self.post_transform(generate_img)
                            self.eval_metrics.update(fake=generate_img_fid.contiguous())

                            if save_img is None:
                                save_img = generate_img_fid

                            if self.save_npz:
                                # convert [0, 1] to [0, 255]
                                data = torch.clamp(255 * generate_img_fid, 0, 255)
                                data = (
                                    rearrange(data, "b c h w -> b h w c")
                                    .to("cpu", dtype=torch.uint8)
                                    .numpy()
                                )
                                name = f"{name}_rank{dist.get_rank()}_iter{i}"
                                npz_path = os.path.join(npy_proc, f"{name}.npy")
                                np.save(npz_path, arr=data)
                    # break

                # save image for checking training
                dist.barrier()
                if self.is_main_process:
                    save_image(
                        make_grid(
                            torch.cat([save_img[:16]]),
                            nrow=8,
                        ),
                        os.path.join(self.save, f"samples/{name}.jpg"),
                        # normalize=True,
                    )
                dist.barrier()

                eval_results = self.eval_metrics.compute_and_reduce()
                eval_results = {f"{name}": eval_results.get("fid", -1)}
                logging_info(eval_results)

                eval_results_total = update_dict(eval_results_total, eval_results)

                if self.is_main_process and self.save_npz:
                    data_list = []
                    num = 0
                    for file in os.listdir(npy_proc):
                        data = np.load(os.path.join(npy_proc, file))
                        data_list.append(data)
                        num += data.shape[0]

                    data = np.concatenate(data_list, axis=0)
                    assert data.shape == (num, data.shape[1], data.shape[2], 3)
                    npz_path = os.path.join(self.save, f"{name}.npz")
                    np.savez(npz_path, arr_0=data)

            self.logger.log(**(eval_results_total))
