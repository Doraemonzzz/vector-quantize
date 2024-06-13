import os

import torch
import torch.distributed as dist
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders_by_args
from vector_quants.loss import LPIPS, get_revd_perceptual
from vector_quants.models import VQVAE
from vector_quants.utils import get_args, logging_info, multiplyList, type_dict


def main():
    args = get_args()
    distributed.enable(overwrite=True)
    args.distributed = True
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # 1, load dataset
    val_data_loader = get_data_loaders_by_args(args, is_train=False)
    transform_rev = transforms.Normalize(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    )

    # 2, load model
    model = VQVAE(args)
    model.cuda(torch.cuda.current_device())
    ckpt_dir = os.path.join(args.save, args.load)
    logging_info(f"Load from {ckpt_dir}")
    state_dict = torch.load(ckpt_dir, map_location="cpu")["model_state_dict"]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    dtype = type_dict[args.dtype]
    model.load_state_dict(new_state_dict)
    # ddp
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True
    )
    model.eval()

    # load perceptual model
    perceptual_model = LPIPS().eval()
    perceptual_model.cuda(torch.cuda.current_device())

    get_l1_loss = torch.nn.L1Loss()
    # for FID
    fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda(
        torch.cuda.current_device()
    )
    # for compute codebook usage
    num_embed = multiplyList(args.levels)
    codebook_usage = set()

    total_l1_loss = 0
    total_per_loss = 0
    num_iter = 0

    torch.distributed.barrier()

    for input_img, _ in tqdm(val_data_loader):
        # forward
        num_iter += 1
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                input_img = input_img.cuda(torch.cuda.current_device())
                reconstructions, codebook_loss, ids = model(input_img, return_id=True)
            # save_image(make_grid(torch.cat([input_img, reconstructions]), nrow=input_img.shape[0]), 'figures/' + str(num_embed)+'.jpg', normalize=True)
            # exit()
            ids = torch.flatten(ids)
            for quan_id in ids:
                codebook_usage.add(quan_id.item())

        # compute L1 loss and perceptual loss
        perceptual_loss = get_revd_perceptual(
            input_img, reconstructions, perceptual_model
        )
        l1loss = get_l1_loss(input_img, reconstructions)
        total_l1_loss += l1loss.cpu().item()
        total_per_loss += perceptual_loss.cpu().item()
        input_img = transform_rev(input_img.contiguous())
        reconstructions = transform_rev(reconstructions.contiguous())

        fid.update(input_img, real=True)
        fid.update(reconstructions, real=False)

    fid_score = fid.compute().item()
    # summary result
    world_size = torch.distributed.get_world_size()
    loss = torch.Tensor([fid_score, total_l1_loss, total_per_loss]).cuda()
    codebook_usage_list = [None for _ in range(world_size)]
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    dist.all_gather_object(codebook_usage_list, codebook_usage)
    loss /= world_size
    codebook_usage = set()
    for codebook_usange_ in codebook_usage_list:
        codebook_usage = codebook_usage.union(codebook_usange_)

    logging_info(f"fid score: {loss[0].item()}")
    logging_info(f"l1loss: {loss[1] / num_iter}")
    logging_info(f"precep_loss: {loss[2] / num_iter}")
    logging_info(f"codebook usage: {len(codebook_usage) / num_embed}")


if __name__ == "__main__":
    main()
