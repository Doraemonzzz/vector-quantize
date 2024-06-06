import logging
import os

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
    model = VQVAE(args)

    dtype = type_dict[args.dtype]
    logging_info(model)

    logging_info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )
    logging_info(f"Dtype {dtype}")
    model.cuda(torch.cuda.current_device())
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    scaler = torch.cuda.amp.GradScaler()

    # 3, load optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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

    torch.distributed.barrier()
    # 5. begin training
    num_iter = 0
    get_l1loss = torch.nn.L1Loss()

    for epoch in range(args.max_train_epochs):
        train_data_loader.sampler.set_epoch(epoch)
        for _, (input_img, _) in enumerate(train_data_loader):
            num_iter += 1
            # test saving
            if num_iter == 1:
                torch.save(
                    {
                        "iter": num_iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict(),
                    },
                    args.save + "/ckpts/{}.pt".format(epoch),
                )

            # forward
            input_img = input_img.cuda(torch.cuda.current_device())
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                reconstructions, codebook_loss, _ = model(input_img)
                l1loss = get_l1loss(input_img, reconstructions)
                perceptual_loss = get_revd_perceptual(
                    input_img, reconstructions, perceptual_model
                )
                loss = codebook_loss + l1loss + perceptual_loss

            if use_wandb and is_main_process():
                wandb.log(
                    {
                        "codebook_loss": codebook_loss.item(),
                        "l1loss": l1loss.item(),
                        "perceptual_loss": perceptual_loss.item(),
                    }
                )

            # # backward
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # lr_scheduler.step()

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

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
            epoch % 5 == 0 or (epoch == args.max_train_epochs - 1)
        ) and is_main_process() == 0:
            torch.save(
                {
                    "iter": num_iter,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                },
                args.save + "/ckpts/{}.pt".format(epoch),
            )


if __name__ == "__main__":
    main()
