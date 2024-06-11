import datetime
import os
import time
from collections import OrderedDict

import torch
import wandb
from torchvision.utils import make_grid, save_image

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.loss import Loss, get_post_transform
from vector_quants.models import get_model
from vector_quants.scheduler import AnnealingLR
from vector_quants.utils import (
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    set_random_seed,
    type_dict,
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

        # other params
        self.max_train_epochs = args.max_train_epochs
        self.log_interval = args.log_interval
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.save_interval = args.save_interval
        self.save = args.save
        self.post_transform = get_post_transform(args.post_transform_type)

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
                    # logging_info(
                    #     "rank 0: epoch:{}, iter:{}, lr:{:.4}, l1loss:{:.4}, percep_loss:{:.4}, codebook_loss:{:.4}".format(
                    #         epoch,
                    #         num_iter,
                    #         optimizer.state_dict()["param_groups"][0]["lr"],
                    #         l1loss.item(),
                    #         perceptual_loss.item(),
                    #         codebook_loss.item(),
                    #     )
                    # )
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
        return 0
        # for input_img, _ in tqdm(self.val_data_loader, disable=not self.is_main_process):
        #     # forward
        #     num_iter += 1
        #     with torch.no_grad():
        #         with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
        #             input_img = input_img.cuda(torch.cuda.current_device())
        #             reconstructions, codebook_loss, ids = self.model(input_img, return_id=True)

        #         ids = torch.flatten(ids)
        #         for quan_id in ids:
        #             codebook_usage.add(quan_id.item())

        #     input_img, reconstructions = self.post_transform(input_img), self.post_transform(reconstructions)
        #     # compute L1 loss and perceptual loss
        #     loss, loss_dict = self.loss_fn(input_img, reconstructions)
        #     total_l1_loss += l1loss.cpu().item()
        #     total_per_loss += perceptual_loss.cpu().item()

        #     fid.update(input_img, real=True)
        #     fid.update(reconstructions, real=False)

        # fid_score = fid.compute().item()
        # # summary result
        # world_size = torch.distributed.get_world_size()
        # loss = torch.Tensor([fid_score, total_l1_loss, total_per_loss]).cuda()
        # codebook_usage_list = [None for _ in range(world_size)]
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        # dist.all_gather_object(codebook_usage_list, codebook_usage)
        # loss /= world_size
        # codebook_usage = set()
        # for codebook_usange_ in codebook_usage_list:
        #     codebook_usage = codebook_usage.union(codebook_usange_)

        # logging_info(f"fid score: {loss[0].item()}")
        # logging_info(f"l1loss: {loss[1] / num_iter}")
        # logging_info(f"precep_loss: {loss[2] / num_iter}")
        # logging_info(f"codebook usage: {len(codebook_usage) / num_embed}")
