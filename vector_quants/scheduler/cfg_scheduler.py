# reference:
#     https://github.com/Pepper-lll/LMforImageGeneration/blob/main/llama/ar_model.py
#     https://github.com/LTH14/mar/blob/main/models/mar.py

import numpy as np


class CfgScheduler:
    def __init__(self, cfg_scale=1, num_steps=256, cfg_schedule="constant"):
        super().__init__()
        self.cfg_scale = cfg_scale
        self.num_steps = num_steps
        self.cfg_schedule = cfg_schedule
        self.step = 1

        assert self.cfg_schedule in [
            "linear",
            "cos",
            "log",
            "square",
            "square_root",
            "constant",
        ], f"Invalid cfg_schedule: {self.cfg_schedule}"

    def reset(self, cfg_scale=None, cfg_schedule=None):
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale
        if cfg_schedule is not None:
            self.cfg_schedule = cfg_schedule
        self.step = 1

    def get_cfg(self):
        ratio = 1 + self.step / self.num_steps
        if self.cfg_schedule == "linear":
            cfg = 1.0 + (self.cfg_scale - 1.0) * ratio
        elif self.cfg_schedule == "cos":
            theta = -np.pi / 2 + ratio * (np.pi / 2)
            cfg = 1.0 + (self.cfg_scale - 1.0) * np.cos(theta)
        elif self.cfg_schedule == "log":
            cfg = 1.0 + (self.cfg_scale - 1.0) * np.log1p(
                ratio * (np.e - 1.0)
            ) / np.log1p(np.e - 1.0)
        elif self.cfg_schedule == "square":
            cfg = 1.0 + (self.cfg_scale - 1) * ratio**2
        elif self.cfg_schedule == "square_root":
            ratio = 1.0 * (step) / num_iter
            cfg = 1.0 + (self.cfg_scale - 1.0) * (ratio**0.5)
        elif self.cfg_schedule == "constant":
            cfg = self.cfg_scale
        else:
            cfg = self.cfg_scale

        self.step += 1
        if self.step > self.num_steps:
            self.reset()

        return cfg
