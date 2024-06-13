import datetime
import os
import time
from collections import OrderedDict

import torch
import wandb
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.loss import Loss, get_post_transform
from vector_quants.metrics import CodeBookMetric, Metrics
from vector_quants.models import get_model
from vector_quants.scheduler import AnnealingLR
from vector_quants.utils import (
    get_metrics_list,
    get_num_embed,
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    print_dict,
    reduce_dict,
    set_random_seed,
    type_dict,
    update_dict,
)

from .base_trainer import BaseTrainer


class VQTrainer(BaseTrainer):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        set_random_seed(args.seed)

        distributed.enable(overwrite=True)
        args.distributed = True
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

        mkdir_ckpt_dirs(args)

        # setup wandb
        self.use_wandb = args.use_wandb
        if self.use_wandb and self.is_main_process:
            wandb_cache_dir = os.path.join(args.wandb_cache_dir, args.wandb_exp_name)
            os.makedirs(wandb_cache_dir, exist_ok=True)
            logging_info(f"wandb will be saved at {wandb_cache_dir}")
            wandb.init(
                config=args,
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=args.wandb_exp_name,
                dir=wandb_cache_dir,
            )

        # 1, load dataset
        self.train_data_loader = get_data_loaders(args, is_train=True)
        self.val_data_loader = get_data_loaders(args, is_train=False)

        # 2, load model
        self.model = get_model(args)

        self.dtype = type_dict[args.dtype]
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.lr_scheduler = AnnealingLR(
            self.optimizer,
            start_lr=args.lr,
            warmup_iter=args.warmup * args.train_iters,
            num_iters=args.train_iters,
            decay_style=args.lr_decay_style,
            last_iter=-1,
            decay_ratio=args.lr_decay_ratio,
        )

        # 4. get loss
        self.loss_fn = Loss(
            perceptual_loss_type=args.perceptual_loss_type,
            adversarial_loss_type=args.adversarial_loss_type,
            l1_loss_weight=args.l1_loss_weight,
            l2_loss_weight=args.l2_loss_weight,
            perceptual_loss_weight=args.perceptual_loss_weight,
            adversarial_loss_weight=args.adversarial_loss_weight,
            codebook_loss_weight=args.codebook_loss_weight,
        )
        torch.distributed.barrier()

        # 5. resume
        self.start_epoch, self.num_iter = self.resume(args.ckpt_path)
        logging_info(self.start_epoch)

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[args.gpu], find_unused_parameters=True
        )

        # evaluation
        self.eval_metrics = Metrics(
            metrics_list=get_metrics_list(args.metrics_list),
            dataset_name=args.data_set,
            device=torch.cuda.current_device(),
        )
        self.codebook_metric = CodeBookMetric(get_num_embed(args))
        self.num_embed = get_num_embed(args)

        # other params
        self.max_train_epochs = args.max_train_epochs
        self.log_interval = args.log_interval
        self.eval_interval = args.eval_interval
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.save_interval = args.save_interval
        self.save = args.save
        self.post_transform = get_post_transform(
            args.post_transform_type, data_set=args.data_set
        )

    @property
    def is_main_process(self):
        return is_main_process()

    def resume(self, ckpt_path):
        if ckpt_path == None:
            logging_info(f"Train from scratch")
            return 0, 0

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

        for epoch in range(start_epoch, self.max_train_epochs):
            self.train_data_loader.sampler.set_epoch(epoch)

            if (epoch + 1) % self.eval_interval == 0:
                # if epoch % self.eval_interval == 0:
                self.eval()

            self.model.train()

            for _, (input_img, _) in enumerate(self.train_data_loader):
                # test saving
                if num_iter == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "iter": num_iter,
                            "model_state_dict": self.model.module.state_dict(),  # remove ddp
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.lr_scheduler.state_dict(),
                            "scaler_state_dict": self.scaler.state_dict(),
                        },
                        os.path.join(self.save, f"ckpts/{epoch}.pt"),
                    )

                # forward
                input_img = input_img.cuda(torch.cuda.current_device())
                with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                    reconstructions, codebook_loss, _ = self.model(input_img)
                    loss, loss_dict = self.loss_fn(
                        codebook_loss,
                        self.post_transform(input_img),
                        self.post_transform(reconstructions),
                    )

                if self.use_wandb and self.is_main_process:
                    wandb.log(loss_dict)

                # backward
                self.scaler.scale(loss).backward()
                if num_iter % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    num_iter += 1

                # print info
                if self.is_main_process and num_iter % self.log_interval == 0:
                    lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                    info = f"rank 0: epoch: {epoch}, iter: {num_iter}, lr: {lr}, "
                    for key in loss_dict:
                        info += f"{key}: {loss_dict[key]}, "
                    logging_info(info)

                # save image for checking training
                if self.is_main_process and num_iter % self.log_interval == 0:
                    save_image(
                        make_grid(
                            torch.cat([input_img, reconstructions]),
                            nrow=input_img.shape[0],
                        ),
                        os.path.join(self.save, "samples/{num_iter}.jpg"),
                        normalize=True,
                    )

            # save checkpoints
            if (
                epoch % self.save_interval == 0 or (epoch == self.max_train_epochs - 1)
            ) and self.is_main_process == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,  # next epoch
                        "iter": num_iter,
                        "model_state_dict": self.model.module.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.lr_scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                    },
                    os.path.join(self.save, f"ckpts/{epoch}.pt"),
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging_info("Training time {}".format(total_time_str))

        self.eval()

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
                    reconstructions, codebook_loss, indices = self.model(
                        input_img, return_id=True
                    )
                    loss, loss_dict = self.loss_fn(
                        codebook_loss,
                        self.post_transform(input_img),
                        self.post_transform(reconstructions),
                    )

                loss_dict_total = update_dict(loss_dict_total, loss_dict)

            self.eval_metrics.update(
                real=input_img.contiguous(), fake=reconstructions.contiguous()
            )
            self.codebook_metric.update(indices)

        eval_loss_dict = reduce_dict(loss_dict_total, prefix="valid_")
        eval_results = self.eval_metrics.compute_and_reduce()
        codebook_results = self.codebook_metric.get_result()

        print_dict(eval_loss_dict)
        print_dict(eval_results)
        print_dict(codebook_results)

        if self.use_wandb and self.is_main_process:
            wandb.log(eval_loss_dict)
            wandb.log(eval_results)
            wandb.log(codebook_results)

        torch.cuda.empty_cache()
        logging_info("End Evaluation")