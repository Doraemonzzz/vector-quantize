import os
from dataclasses import asdict
from pprint import pformat

import torch
from tqdm import tqdm

import vector_quants.utils.distributed as distributed
from vector_quants.data import get_data_loaders
from vector_quants.loss import Loss, get_post_transform
from vector_quants.metrics import CodeBookMetric, Metrics
from vector_quants.models import AutoVqVae
from vector_quants.utils import (
    get_metrics_list,
    get_num_embed,
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

        self.cfg = cfg
        cfg_model = cfg.model
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
        self.model = AutoVqVae.from_pretrained(cfg_train.ckpt_path)

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

        # 3. get loss
        self.loss_fn = Loss(
            perceptual_loss_type=cfg_loss.perceptual_loss_type,
            adversarial_loss_type=cfg_loss.adversarial_loss_type,
            l1_loss_weight=cfg_loss.l1_loss_weight,
            l2_loss_weight=cfg_loss.l2_loss_weight,
            perceptual_loss_weight=cfg_loss.perceptual_loss_weight,
            adversarial_loss_weight=cfg_loss.adversarial_loss_weight,
            codebook_loss_weight=cfg_loss.codebook_loss_weight,
            entropy_loss_weight=cfg_model.entropy_loss_weight,
        )
        torch.distributed.barrier()

        # 4. ddp setup
        num_embed = self.model.num_embed
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[cfg_train.gpu], find_unused_parameters=True
        )

        # evaluation
        self.eval_metrics = Metrics(
            metrics_list=get_metrics_list(cfg_loss.metrics_list),
            dataset_name=cfg_data.data_set,
            device=torch.cuda.current_device(),
        )

        self.num_embed = num_embed if num_embed != -1 else get_num_embed(cfg_model)
        self.codebook_metric = CodeBookMetric(self.num_embed)

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

        logging_info((eval_loss_dict | eval_results | codebook_results))

        torch.cuda.empty_cache()
        logging_info("End Evaluation")
