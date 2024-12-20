import datetime
import os
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from pprint import pformat

import torch
import torch._dynamo
from einops._torch_specific import (  # https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    allow_ops_in_compiled_graph,
)
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import DATASET_CONFIGS, get_data_loaders
from vector_quants.logger import Logger
from vector_quants.loss import Loss, get_post_transform
from vector_quants.metrics import CodeBookMetric, Metrics, metrics_names
from vector_quants.models import AutoVqVae
from vector_quants.optim import get_optimizer
from vector_quants.scheduler import AnnealingLR
from vector_quants.utils import (
    compute_grad_norm,
    compute_num_patch,
    get_metrics_list,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    reduce_dict,
    requires_grad,
    set_random_seed,
    type_dict,
    update_dict,
    update_ema,
)

from .base_trainer import BaseTrainer

allow_ops_in_compiled_graph()
torch._dynamo.config.suppress_errors = True


class VQTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        logging_info(pformat(asdict(cfg)))

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

        # 1, load dataset
        self.train_data_loader = get_data_loaders(cfg_train, cfg_data, is_train=True)
        self.val_data_loader = get_data_loaders(cfg_train, cfg_data, is_train=False)
        cfg_train.train_iters = int(
            DATASET_CONFIGS[cfg_data.data_set]["samples"]
            / cfg_train.batch_size
            / cfg_train.gradient_accumulation_steps
            / cfg_train.world_size
            * cfg_train.max_train_epochs
        )
        logging_info(f"train_iters: {cfg_train.train_iters}")

        # 2, load model
        self.model = AutoVqVae.from_config(cfg_model)

        frozen_blocks = cfg_model.frozen_blocks.split(",")
        logging_info(f"frozen_blocks: {frozen_blocks}")
        for name, child in self.model.named_children():
            if name in frozen_blocks:
                for param in child.parameters():
                    param.requires_grad = False

        self.dtype = type_dict[cfg_train.dtype]
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
        # ema model
        self.ema = cfg_model.ema
        if self.ema:
            self.ema_model = deepcopy(self.model).to(
                torch.cuda.current_device()
            )  # Create an EMA of the model for use after training
            requires_grad(self.ema_model, False)
            logging_info(
                f"VQ Model EMA Parameters: {sum(p.numel() for p in self.ema_model.parameters()):,}"
            )
            update_ema(
                self.ema_model, self.model, decay=0
            )  # Ensure EMA is initialized with synced weights

        self.compile = cfg_train.compile
        if self.compile:
            logging_info("compiling the model... (may take several minutes)")
            self.model = torch.compile(self.model)
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
        if self.compile:
            logging_info("compiling the loss... (may take several minutes)")
            self.loss_fn = torch.compile(self.loss_fn)
        logging_info(self.loss_fn)
        loss_fn_keys = self.loss_fn.keys

        # setup disc optimizer
        self.disc_type = cfg_loss.disc_type
        if self.disc_type != "none":
            self.optimizer_disc = get_optimizer(cfg_train, self.loss_fn.discriminator)
            self.lr_scheduler_disc = AnnealingLR(
                self.optimizer_disc,
                start_lr=cfg_train.lr,
                warmup_iter=cfg_train.warmup * cfg_train.train_iters,
                num_iters=cfg_train.train_iters,
                decay_style=cfg_train.lr_decay_style,
                last_iter=-1,
                decay_ratio=cfg_train.lr_decay_ratio,
            )
            self.scaler_disc = torch.cuda.amp.GradScaler()

            logging_info(
                "num. disc model params: {:,} (num. trained: {:,})".format(
                    sum(
                        getattr(p, "_orig_size", p).numel()
                        for p in self.loss_fn.discriminator.parameters()
                    ),
                    sum(
                        getattr(p, "_orig_size", p).numel()
                        for p in self.loss_fn.discriminator.parameters()
                        if p.requires_grad
                    ),
                )
            )
        else:
            self.optimizer_disc = None
            self.lr_scheduler_disc = None
            self.scaler_disc = None

        self.disc_loss_start_iter = cfg_loss.disc_loss_start_iter

        torch.distributed.barrier()

        # 5. resume
        self.only_load_model = cfg_train.only_load_model
        self.start_epoch, self.num_iter = self.resume(cfg_train.ckpt_path_stage1)
        logging_info(f"Start epoch: {self.start_epoch}")

        num_embed = self.model.num_embed
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[cfg_train.gpu]
        )
        if self.disc_type != "none":
            self.loss_fn = torch.nn.parallel.DistributedDataParallel(
                self.loss_fn, device_ids=[cfg_train.gpu], find_unused_parameters=True
            )

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

        mkdir_ckpt_dirs(cfg_train)

        # other params
        self.cfg = cfg
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
        self.num_sample = cfg_data.num_sample

    @property
    def is_main_process(self):
        return is_main_process()

    def process_state_dict(self, current_model_dict, state_dict):
        miss_match_keys = []
        for k in current_model_dict.keys():
            if (
                k not in state_dict.keys()
                or current_model_dict[k].size() != state_dict[k].size()
            ):
                miss_match_keys.append(k)
        for k in miss_match_keys:
            logging_info(f"Miss match key: {k}")
        new_state_dict = {
            k: v
            if (k not in miss_match_keys and v.size() == current_model_dict[k].size())
            else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), state_dict.values())
        }

        return new_state_dict

    def resume(self, ckpt_path):
        num_iter = 0
        start_epoch = 0
        if ckpt_path == None:
            logging_info(f"Train from scratch")
            return num_iter, start_epoch

        logging_info(f"Only load model: {self.only_load_model}")
        pkg = torch.load(ckpt_path, map_location="cpu")

        ##### g load
        # create new OrderedDict that does not contain `module.`
        state_dict = OrderedDict()
        for k, v in pkg["model_state_dict"].items():
            name = k.replace("module.", "")
            state_dict[name] = v

        # process state dict
        # load params
        new_state_dict = self.process_state_dict(self.model.state_dict(), state_dict)
        model_msg = self.model.load_state_dict(new_state_dict, strict=False)

        if not self.only_load_model:
            self.optimizer.load_state_dict(pkg["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(pkg["scheduler_state_dict"])
            self.scaler.load_state_dict(pkg["scaler_state_dict"])

            num_iter = pkg["iter"]
            start_epoch = pkg["epoch"]

        ##### ema load
        if self.ema:
            ema_state_dict = pkg["ema_model_state_dict"]
            # process state dict
            # load params
            ema_new_state_dict = self.process_state_dict(
                self.ema_model.state_dict(), ema_state_dict
            )
            ema_model_msg = self.ema_model.load_state_dict(
                ema_new_state_dict, strict=False
            )
            logging_info(f"EMA: {ema_model_msg}")

        ##### d load
        if (
            self.disc_type != "none" and pkg["model_disc_state_dict"] is not None
        ):  # add this to support pretraining
            # create new OrderedDict that does not contain `module.`
            state_disc_dict = OrderedDict()
            for k, v in pkg["model_disc_state_dict"].items():
                name = k.replace("module.", "")
                state_disc_dict[name] = v
            # load params
            model_disc_msg = self.loss_fn.discriminator.load_state_dict(state_disc_dict)

            self.optimizer_disc.load_state_dict(pkg["optimizer_disc_state_dict"])
            self.lr_scheduler_disc.load_state_dict(pkg["scheduler_disc_state_dict"])
            self.scaler_disc.load_state_dict(pkg["scaler_disc_state_dict"])
            logging_info(f"Discriminator: {model_disc_msg}")

        # logging
        logging_info(f"Load from {ckpt_path}")
        logging_info(f"Generator: {model_msg}")
        logging_info(f"Resume from epoch {start_epoch}, iter {num_iter}")

        return start_epoch, num_iter

    def train(self):
        start_time = time.time()

        start_epoch = self.start_epoch
        num_iter = self.num_iter

        for epoch in range(start_epoch, self.max_train_epochs):
            self.train_data_loader.sampler.set_epoch(epoch)
            # self.eval()

            self.model.train()
            self.loss_fn.train()
            if self.ema:
                self.ema_model.eval()

            for _, (input_img, _) in enumerate(self.train_data_loader):
                # test saving
                if num_iter == 0 and self.is_main_process:
                    if self.disc_type != "none":
                        model_disc_state_dict = (
                            self.loss_fn.module._orig_mod.discriminator.state_dict()
                            if self.compile
                            else self.model.module.state_dict()
                        )
                    else:
                        model_disc_state_dict = None

                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": num_iter,
                            "model_state_dict": self.model.module._orig_mod.state_dict()
                            if self.compile
                            else self.model.module.state_dict(),  # remove ddp
                            "ema_model_state_dict": self.ema_model.state_dict()
                            if self.ema
                            else None,
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.lr_scheduler.state_dict(),
                            "scaler_state_dict": self.scaler.state_dict(),
                            "cfg": self.cfg,
                            # disc
                            "model_disc_state_dict": model_disc_state_dict,
                            "optimizer_disc_state_dict": self.optimizer_disc.state_dict()
                            if self.disc_type != "none"
                            else None,
                            "scheduler_disc_state_dict": self.lr_scheduler_disc.state_dict()
                            if self.disc_type != "none"
                            else None,
                            "scaler_disc_state_dict": self.scaler_disc.state_dict()
                            if self.disc_type != "none"
                            else None,
                        },
                        os.path.join(self.save, f"ckpts/{epoch}.pt"),
                    )
                    logging_info("Finish test saving.")

                ##### update g
                input_img = input_img.cuda(torch.cuda.current_device())
                if (num_iter + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    reconstructions, indices, loss_dict = self.model(input_img)
                    input_img = self.post_transform(input_img)
                    reconstructions = self.post_transform(reconstructions)
                    loss, loss_dict = self.loss_fn(
                        input_img,
                        reconstructions,
                        num_iter=num_iter,
                        **loss_dict,
                    )
                    loss = (
                        loss / self.gradient_accumulation_steps
                    )  # scale the loss to account for gradient accumulation

                # backward
                self.scaler.scale(loss).backward()

                # compute grad norm
                grad_norm = 0
                if self.is_main_process and (num_iter + 1) % self.log_interval == 0:
                    grad_norm = compute_grad_norm(
                        self.model, scale=self.scaler.get_scale()
                    )

                if (num_iter + 1) % self.gradient_accumulation_steps == 0:
                    if self.clip_grad:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    if self.ema:
                        update_ema(
                            self.ema_model,
                            self.model.module._orig_mod
                            if self.compile
                            else self.model.module,
                        )

                ##### update d
                grad_disc_norm = 0
                if self.use_disc(num_iter):
                    if (num_iter + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer_disc.zero_grad()  # !!!!!! important

                    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                        loss_disc, loss_disc_dict = self.loss_fn(
                            input_img,
                            reconstructions,
                            num_iter=num_iter,
                            is_disc=True,
                            scale=self.scaler_disc.get_scale(),
                        )
                        loss_disc = (
                            loss_disc / self.gradient_accumulation_steps
                        )  # scale the loss to account for gradient accumulation
                    # backward
                    self.scaler_disc.scale(loss_disc).backward()

                    # disc
                    if self.is_main_process and (num_iter + 1) % self.log_interval == 0:
                        grad_disc_norm = compute_grad_norm(
                            self.loss_fn.module.discriminator,
                            scale=self.scaler_disc.get_scale(),
                        )

                    if (num_iter + 1) % self.gradient_accumulation_steps == 0:
                        if self.clip_grad:
                            # disc
                            self.scaler_disc.unscale_(self.optimizer_disc)
                            torch.nn.utils.clip_grad_norm_(
                                self.loss_fn.module.discriminator.parameters(),
                                self.clip_grad,
                            )

                        self.scaler_disc.step(self.optimizer_disc)
                        self.scaler_disc.update()
                        self.lr_scheduler_disc.step()
                else:
                    loss_disc_dict = {}

                # print info
                if (num_iter + 1) % (
                    self.log_interval * self.gradient_accumulation_steps
                ) == 0:
                    self.optimizer.state_dict()["param_groups"][0]["lr"]
                    self.logger.log(
                        **(
                            loss_dict
                            | loss_disc_dict
                            | {
                                "epoch": epoch + 1,
                                "iter": num_iter + 1,
                                "lr": self.optimizer.state_dict()["param_groups"][0][
                                    "lr"
                                ],
                                "d_lr": self.optimizer_disc.state_dict()[
                                    "param_groups"
                                ][0]["lr"]
                                if self.disc_type != "none"
                                else 0,
                                "gnorm": grad_norm,
                                "grad_disc_norm": grad_disc_norm,
                            }
                        ),
                    )

                num_iter += 1

            if (epoch + 1) % self.eval_interval == 0:
                self.eval()

            # save checkpoints
            if (
                (epoch + 1) % self.save_interval == 0
                or (epoch == self.max_train_epochs - 1)
            ) and self.is_main_process:
                if self.disc_type != "none":
                    model_disc_state_dict = (
                        self.loss_fn.module._orig_mod.discriminator.state_dict()
                        if self.compile
                        else self.loss_fn.module.discriminator.state_dict()
                    )
                else:
                    model_disc_state_dict = None

                torch.save(
                    {
                        "epoch": epoch + 1,  # next epoch
                        "iter": num_iter,
                        "model_state_dict": self.model.module._orig_mod.state_dict()
                        if self.compile
                        else self.model.module.state_dict(),
                        "ema_model_state_dict": self.ema_model.state_dict()
                        if self.ema
                        else None,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "cfg": self.cfg,
                        # disc
                        "model_disc_state_dict": model_disc_state_dict,
                        "optimizer_disc_state_dict": self.optimizer_disc.state_dict()
                        if self.disc_type != "none"
                        else None,
                        "scheduler_disc_state_dict": self.lr_scheduler_disc.state_dict()
                        if self.disc_type != "none"
                        else None,
                        "scaler_disc_state_dict": self.scaler_disc.state_dict()
                        if self.disc_type != "none"
                        else None,
                    },
                    os.path.join(self.save, f"ckpts/{epoch + 1}.pt"),
                )
                # save image for checking training
                save_image(
                    make_grid(
                        torch.cat([input_img, reconstructions]),
                        nrow=input_img.shape[0],
                    ),
                    os.path.join(self.save, f"samples/epoch_{epoch + 1}.jpg"),
                    normalize=True,
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging_info("Training time {}".format(total_time_str))

        self.eval()

    def use_disc(self, num_iter):
        return self.disc_type != "none" and num_iter >= self.disc_loss_start_iter

    def eval(self):
        logging_info("Start Evaluation")
        self.model.eval()
        self.loss_fn.eval()

        model_list = [self.model]
        prefix_list = [""]
        if self.ema:
            model_list += [self.ema_model]
            prefix_list += ["ema_"]

        for i in range(len(model_list)):
            model = model_list[i]
            prefix = prefix_list[i]
            self.eval_metrics.reset()

            loss_dict_total = {}
            for input_img, _ in tqdm(
                self.val_data_loader, disable=not self.is_main_process
            ):
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                        input_img = input_img.cuda(torch.cuda.current_device())
                        reconstructions, indices, loss_dict = model(input_img)
                        # rescale to [0, 1]
                        input_img = self.post_transform(input_img)
                        reconstructions = self.post_transform(reconstructions)
                        loss, loss_dict = self.loss_fn(
                            input_img,
                            reconstructions,
                            **loss_dict,
                        )

                        if self.use_disc(num_iter=1e5):
                            with torch.amp.autocast(
                                device_type="cuda", dtype=self.dtype
                            ):
                                loss_disc, loss_disc_dict = self.loss_fn(
                                    input_img,
                                    reconstructions,
                                    num_iter=1e5,
                                    is_disc=True,
                                )
                            loss_dict = update_dict(loss_dict, loss_disc_dict)

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

            loss_dict = eval_loss_dict | eval_results | codebook_results
            keys = list(loss_dict.keys())
            for key in keys:
                loss_dict[prefix + key] = loss_dict.pop(key)
            self.logger.log(**loss_dict)

        torch.cuda.empty_cache()
        logging_info("End Evaluation")
