import datetime
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from pprint import pformat

import torch
import torch.nn.functional as F
from einops import pack
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import DATASET_CONFIGS, get_data_loaders
from vector_quants.logger import Logger
from vector_quants.loss import get_post_transform
from vector_quants.metrics import CodeBookMetric, Metrics, metrics_names
from vector_quants.models import AutoArModel, AutoVqVae
from vector_quants.optim import get_optimizer
from vector_quants.scheduler import AnnealingLR
from vector_quants.utils import (
    compute_grad_norm,
    get_metrics_list,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    reduce_dict,
    set_random_seed,
    type_dict,
    update_dict,
)

from .base_trainer import BaseTrainer


class ARTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        logging_info(pformat(asdict(cfg)))

        self.cfg = cfg
        cfg_model_stage2 = cfg.model_stage2
        cfg_train = cfg.train
        cfg_data = cfg.data
        cfg_loss = cfg.loss

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
        self.vqvae = AutoVqVae.from_pretrained(cfg_train.ckpt_path_stage1).to(
            self.dtype
        )
        self.vqvae.cuda(torch.cuda.current_device())
        logging_info(self.vqvae)
        cfg_model_stage2.num_class = DATASET_CONFIGS[cfg_data.data_set]["num_class"]
        cfg_model_stage2.vocab_size = self.vqvae.num_embed
        self.model = AutoArModel.from_config(cfg_model_stage2)

        logging_info(self.model)

        logging_info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(
                    getattr(p, "_orig_size", p).numel() for p in self.model.parameters()
                ),
                sum(
                    getattr(p, "_orig_size", p).numel()
                    for p in self.model.parameters()
                    if p.requires_grad
                ),
            )
        )
        logging_info(f"Dtype {self.dtype}")
        self.model.cuda(torch.cuda.current_device())
        self.scaler = torch.cuda.amp.GradScaler()

        # 3, load optimizer and scheduler
        self.optimizer = get_optimizer(cfg_train, self.model)

        self.lr_scheduler = AnnealingLR(
            self.optimizer,
            start_lr=cfg_train.lr,
            warmup_iter=cfg_train.warmup * cfg_train.train_iters,
            num_iters=cfg_train.train_iters,
            decay_style=cfg_train.lr_decay_style,
            last_iter=-1,
            decay_ratio=cfg_train.lr_decay_ratio,
        )

        # 4. get loss
        self.loss_fn = F.cross_entropy
        torch.distributed.barrier()

        # 5. resume
        self.start_epoch, self.num_iter = self.resume(cfg_train.ckpt_path_stage2)
        logging_info(f"Start epoch: {self.start_epoch}")

        num_embed = self.model.num_embed
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[cfg_train.gpu], find_unused_parameters=True
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
        )

        self.num_embed = num_embed
        self.codebook_metric = CodeBookMetric(self.num_embed)

        # logger
        self.logger = Logger(
            keys=["epoch", "iter", "lr"]
            + ["cross_entropy"]
            + metrics_names
            + ["gnorm"],
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
        self.post_transform = get_post_transform(
            cfg_loss.post_transform_type, data_set=cfg_data.data_set
        )
        self.clip_grad = cfg_train.clip_grad

    @property
    def is_main_process(self):
        return is_main_process()

    def resume(self, ckpt_path):
        if ckpt_path == None:
            logging_info(f"Train from scratch")
            return 1, 1

        pkg = torch.load(ckpt_path, map_location="cpu")

        # create new OrderedDict that does not contain `module.`
        state_dict = OrderedDict()
        for k, v in pkg["model_state_dict"].items():
            name = k.replace("module.", "")
            state_dict[name] = v
        # load params
        model_msg = self.model.load_state_dict(state_dict)

        self.optimizer.load_state_dict(pkg["optimizer_state_dict"])

        self.lr_scheduler.load_state_dict(pkg["scheduler_state_dict"])

        self.scaler.load_state_dict(pkg["scaler_state_dict"])

        num_iter = pkg["iter"]
        start_epoch = pkg["epoch"]

        # logging
        logging_info(f"Load from {ckpt_path}")
        logging_info(model_msg)
        logging_info(f"Resume from epoch {start_epoch}, iter {num_iter}")

        return start_epoch, num_iter

    def train(self):
        start_time = time.time()

        start_epoch = self.start_epoch
        num_iter = self.num_iter
        self.vqvae.eval()

        for epoch in range(start_epoch, self.max_train_epochs):
            self.train_data_loader.sampler.set_epoch(epoch)

            if epoch % self.eval_interval == 0:
                self.eval()

            self.model.train()

            for _, (input_img, input_label) in enumerate(self.train_data_loader):
                # test saving
                if num_iter == 1 and self.is_main_process:
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": num_iter,
                            "model_state_dict": self.model.module.state_dict(),  # remove ddp
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.lr_scheduler.state_dict(),
                            "scaler_state_dict": self.scaler.state_dict(),
                            "cfg": self.cfg,
                        },
                        os.path.join(self.save, f"ckpts/{epoch}.pt"),
                    )
                    logging_info("Finish test saving.")

                # forward
                input_img = input_img.cuda(torch.cuda.current_device())
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    with torch.no_grad():
                        indices = self.vqvae.encode(input_img)
                        indices, ps = pack([indices], "b *")

                    logits, past_key_values = self.model(indices, input_label)

                    loss = self.loss_fn(
                        logits.view(-1, logits.shape[-1]), indices.view(-1)
                    )
                    loss_dict = {
                        "cross_entropy": loss.item(),
                    }

                # backward
                self.scaler.scale(loss).backward()

                # compute grad norm
                grad_norm = 0
                if self.is_main_process and num_iter % self.log_interval == 0:
                    grad_norm = compute_grad_norm(
                        self.model, scale=self.scaler.get_scale()
                    )

                if num_iter % self.gradient_accumulation_steps == 0:
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # print info
                if num_iter % self.log_interval == 0:
                    self.logger.log(
                        **(
                            loss_dict
                            | {
                                "epoch": epoch,
                                "iter": num_iter,
                                "lr": self.optimizer.state_dict()["param_groups"][0][
                                    "lr"
                                ],
                                "gnorm": grad_norm,
                            }
                        ),
                    )

                num_iter += 1

            # save checkpoints
            if (
                epoch % self.save_interval == 0 or (epoch == self.max_train_epochs - 1)
            ) and self.is_main_process:
                torch.save(
                    {
                        "epoch": epoch + 1,  # next epoch
                        "iter": num_iter,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "cfg": self.cfg,
                    },
                    os.path.join(self.save, f"ckpts/{epoch}.pt"),
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging_info("Training time {}".format(total_time_str))

        self.eval()

    # update this later
    def eval(self):
        logging_info("Start Evaluation")
        self.model.eval()
        self.loss_fn.eval()
        self.eval_metrics.reset()

        loss_dict_total = {}
        for input_img, _ in tqdm(
            self.val_data_loader, disable=not self.is_main_process
        ):
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    input_img = input_img.cuda(torch.cuda.current_device())
                    reconstructions, indices, loss_dict = self.model(
                        input_img, return_id=True
                    )
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
            self.codebook_metric.update(indices)

        eval_loss_dict = reduce_dict(
            loss_dict_total, n=len(self.val_data_loader), prefix="valid_"
        )
        eval_results = self.eval_metrics.compute_and_reduce()
        codebook_results = self.codebook_metric.get_result()

        self.logger.log(**(eval_loss_dict | eval_results | codebook_results))

        torch.cuda.empty_cache()
        logging_info("End Evaluation")
