import datetime
import logging
import os
import time

import torch
import wandb
from torchvision.utils import make_grid, save_image

import distributed
from arguments import get_args
from dataset import get_data_loaders
from lpips import LPIPS
from metric import get_revd_perceptual
from model import VQVAE
from scheduler import AnnealingLR
from utils import (
    is_main_process,
    logging_info,
    mkdir_ckpt_dirs,
    set_random_seed,
    type_dict,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vq")

def resume(model, optimizer, lr_scheduler, scaler, ckpt_path=None):
    if ckpt_path == None:
        logging_info(f"Train from scratch")
        return 0, 0
    
    pkg = torch.load(ckpt_path, map_location="cpu")
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    state_dict = OrderedDict()
    for k, v in pkg["model_state_dict"].items():
        name = k.replace("module.", "")
        state_dict[name] = v
    # load params
    model_msg = model.load_state_dict(state_dict)
    
    
    opt_msg = optimizer.load_state_dict(pkg["optimizer_state_dict"])
    
    scheduler_msg = lr_scheduler.load_state_dict(pkg["scheduler_state_dict"])
    
    
    scaler_msg = scaler.load_state_dict(pkg["scaler_state_dict"])
    
    num_iter = pkg["iter"]
    start_epoch = pkg["epoch"]
    
    # logging
    logging_info(f"Load from {ckpt_path}")
    logging_info(model_msg)
    logging_info(f"Resume from epoch {start_epoch}, iter {num_iter}")
    
    return start_epoch, num_iter

def main():
    args = get_args()
    # initialize_distributed(args)
    distributed.enable(overwrite=True)
    args.distributed = True
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # setup wandb
    use_wandb = args.use_wandb
    if use_wandb and is_main_process():
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

    logging_info(args)

    set_random_seed(args.seed)
    mkdir_ckpt_dirs(args)

    # 1, load dataset
    train_data_loader, val_data_loader = get_data_loaders(args)

    # 2, load model
    model_without_ddp = VQVAE(args)

    dtype = type_dict[args.dtype]
    logging_info(model_without_ddp)

    logging_info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model_without_ddp.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model_without_ddp.parameters()
                if p.requires_grad
            ),
        )
    )
    logging_info(f"Dtype {dtype}")
    model_without_ddp.cuda(torch.cuda.current_device())
    scaler = torch.cuda.amp.GradScaler()

    # 3, load optimizer and scheduler
    optimizer = torch.optim.Adam(
        model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup * args.train_iters,
        num_iters=args.train_iters,
        decay_style=args.lr_decay_style,
        last_iter=-1,
        decay_ratio=args.lr_decay_ratio,
    )

    # 4. load perceptual model
    perceptual_model = LPIPS().eval()
    perceptual_model.cuda(torch.cuda.current_device())
    loss_type = args.loss_type
    if loss_type == 1:
        logging_info("Use codebook loss, l1 loss, perceptual loss")
    elif loss_type == 2:
        logging_info("Use codebook loss, l1 loss")

    torch.distributed.barrier()
    # 5. begin training
    start_epoch, num_iter = resume(model_without_ddp, optimizer, lr_scheduler, scaler, args.ckpt_path)
    get_l1loss = torch.nn.L1Loss()
    
    # ddp
    model = torch.nn.parallel.DistributedDataParallel(
        model_without_ddp, device_ids=[args.gpu], find_unused_parameters=True
    )

    start_time = time.time()

    for epoch in range(start_epoch, args.max_train_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        for _, (input_img, _) in enumerate(train_data_loader):
            # test saving
            if num_iter == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "iter": num_iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                    },
                    args.save + "/ckpts/{}.pt".format(epoch),
                )

            # forward
            input_img = input_img.cuda(torch.cuda.current_device())
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                reconstructions, codebook_loss, _ = model(input_img)
                if loss_type == 1:
                    l1loss = get_l1loss(input_img, reconstructions)
                    perceptual_loss = get_revd_perceptual(
                        input_img, reconstructions, perceptual_model
                    )
                elif loss_type == 2:
                    l1loss = get_l1loss(input_img, reconstructions)
                    perceptual_loss = torch.tensor(0.0).cuda().float()

                loss = codebook_loss + l1loss + perceptual_loss

            if use_wandb and is_main_process():
                wandb.log(
                    {
                        "codebook_loss": codebook_loss.item(),
                        "l1loss": l1loss.item(),
                        "perceptual_loss": perceptual_loss.item(),
                    }
                )

            # backward
            scaler.scale(loss).backward()
            if num_iter % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                num_iter += 1

            # print info
            if is_main_process() and num_iter % args.log_interval == 0:
                logging_info(
                    "rank 0: epoch:{}, iter:{}, lr:{:.4}, l1loss:{:.4}, percep_loss:{:.4}, codebook_loss:{:.4}".format(
                        epoch,
                        num_iter,
                        optimizer.state_dict()["param_groups"][0]["lr"],
                        l1loss.item(),
                        perceptual_loss.item(),
                        codebook_loss.item(),
                    )
                )

            # save image for checking training
            if is_main_process() and num_iter % args.log_interval == 0:
                save_image(
                    make_grid(
                        torch.cat([input_img, reconstructions]), nrow=input_img.shape[0]
                    ),
                    args.save + "/samples/{}.jpg".format(num_iter),
                    normalize=True,
                )

        # save checkpoints
        if (
            epoch % args.save_interval == 0 or (epoch == args.max_train_epochs - 1)
        ) and is_main_process() == 0:
            torch.save(
                {
                    "epoch": epoch, # next epoch
                    "iter": num_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                args.save + "/ckpts/{}.pt".format(epoch),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    main()
