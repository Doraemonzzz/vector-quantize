from torch import nn


class SRmsNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        return x_normed
