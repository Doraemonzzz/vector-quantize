import os

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms
from tqdm import tqdm

from arguments import get_args
from dataset import get_transform
from lpips import LPIPS
from metric import get_revd_perceptual
from model import VQVAE
from utils import multiplyList, type_dict


def main():
    args = get_args()

    # assert args.quantizer == "fsq"

    # 1, load dataset
    imagenet_transform = get_transform(args)
    val_set = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), imagenet_transform
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )
    transform_rev = transforms.Normalize(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225],
    )

    # 2, load model
    model = VQVAE(args)
    model.cuda(torch.cuda.current_device())
    # original saved file with DataParallel
    state_dict = torch.load(args.load)["model_state_dict"]
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    dtype = type_dict[args.dtype]
    model.load_state_dict(new_state_dict)

    model.eval()
    # load perceptual model
    perceptual_model = LPIPS().eval()  # .to(dtype)
    perceptual_model.cuda(torch.cuda.current_device())

    get_l1_loss = torch.nn.L1Loss()
    # for FID
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    # for compute codebook usage
    num_embed = multiplyList(args.levels)
    codebook_usage = set()

    total_l1_loss = 0
    total_per_loss = 0
    num_iter = 0

    for input_img, _ in tqdm(val_data_loader):
        # forward
        num_iter += 1
        print(num_iter * args.batch_size)

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

        fid.update(input_img.cpu(), real=True)
        fid.update(reconstructions.cpu(), real=False)

    print("fid score", fid.compute().item())
    print("l1loss:", total_l1_loss / num_iter)
    print("precep_loss:", total_per_loss / num_iter)
    print("codebook usage", len(codebook_usage) / num_embed)


if __name__ == "__main__":
    main()
