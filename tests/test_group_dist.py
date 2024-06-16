import pytest
import torch

from vector_quants.quantizers.utils import compute_dist, pack_one, unpack_one
from einops import rearrange

n = 4
m = 32


def get_shape(n, m):
    # generate shape up to n dim with max size = m
    shape = []
    for _ in range(1, n + 1):
        shape.append(torch.randint(low=0, high=m, size=(_,)).tolist())

    return shape


@pytest.mark.parametrize(
    "shape",
    get_shape(n, m),
)
@pytest.mark.parametrize(
    "embed_dim",
    [128, 256, 768],
)
@pytest.mark.parametrize(
    "num_embed",
    [1024, 2048],
)
@pytest.mark.parametrize(
    "num_group",
    [1, 2, 4, 8]
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
def test_dist(shape, embed_dim, num_embed, num_group, dtype):
    x = torch.randn(shape + [embed_dim], dtype=dtype).cuda()
    y = torch.randn(num_embed, embed_dim // num_group, dtype=dtype).cuda()

    x_flatten, ps = pack_one(x, "* d")
    x_flatten = rearrange(x_flatten, "... (g d) -> (... g) d", g=num_group)
    dist = compute_dist(x_flatten, y)
    indices = torch.argmin(dist, dim=-1)
    indices = rearrange(indices, "(b g) -> b g", g=num_group)
    indices = unpack_one(indices, ps, "* g")

    # groud truth
    x_flatten_ref = x.reshape(-1, embed_dim)
    x_flatten_ref = rearrange(x_flatten_ref, "... (g d) -> (... g) d", g=num_group)
    dist_ref = (
        x_flatten_ref.pow(2).sum(1, keepdim=True)
        + y.pow(2).sum(1, keepdim=True).t()
        - 2 * x_flatten_ref @ y.t()
    )
    indices_ref = torch.argmin(dist_ref, dim=-1)
    print(indices_ref.shape)
    indices_ref = rearrange(indices_ref, "(b g) -> b g", g=num_group)
    print(indices_ref.shape, shape)
    indices_ref_list = []
    for i in range(num_group):
        indices_ref_list.append(indices_ref[..., i].view(shape).unsqueeze(-1))
    indices_ref = torch.cat(indices_ref_list, dim=-1)
    print(indices_ref.shape)
    # indices_ref = indices_ref.view(shape)

    torch.testing.assert_close(dist_ref, dist)
    torch.testing.assert_close(indices_ref, indices)
