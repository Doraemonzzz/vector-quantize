import os

import wandb

from vector_quants.utils import is_main_process, logging_info

from .gpu_logger import build_gpu_memory_monitor


class Logger:
    def __init__(
        self,
        keys=[],
        use_wandb=False,
        cfg=None,
        wandb_entity=None,
        wandb_project=None,
        wandb_exp_name=None,
        wandb_cache_dir=None,
    ):
        self.gpu_keys = ["max_reserved_gib", "max_reserved_pct"]
        self.keys = keys + self.gpu_keys
        self.use_wandb = use_wandb
        if self.use_wandb and self.is_main_process:
            wandb_cache_dir = os.path.join(wandb_cache_dir, wandb_exp_name)
            os.makedirs(wandb_cache_dir, exist_ok=True)
            logging_info(f"wandb will be saved at {wandb_cache_dir}")
            wandb.init(
                config=cfg,
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_exp_name,
                dir=wandb_cache_dir,
            )

        self.gpu_memory_monitor = build_gpu_memory_monitor()
        self.gpu_memory_monitor.reset_peak_stats()

    @property
    def is_main_process(self):
        return is_main_process()

    def log(self, **kwargs):
        gpu_mem_stats = self.get_gpu_stat()

        if self.is_main_process:
            stat = kwargs | gpu_mem_stats
            logging_info(self.to_string(**stat))
            if self.use_wandb:
                wandb.log(stat)

    def get_gpu_stat(self):
        gpu_mem_stats = self.gpu_memory_monitor.get_peak_stats()

        return {
            "max_reserved_gib": gpu_mem_stats.max_reserved_gib,
            "max_reserved_pct": gpu_mem_stats.max_reserved_pct,
        }

    def to_string(self, **kwargs):
        string = ""
        for key in self.keys:
            if key in kwargs:
                if isinstance(kwargs[key], float):
                    if key in self.gpu_keys:
                        string += f"{key}: {kwargs[key]:.2f}, "
                    else:
                        string += f"{key}: {kwargs[key]:.2e}, "
                else:
                    string += f"{key}: {kwargs[key]}, "

        return string
