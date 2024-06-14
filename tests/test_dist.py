import pytest
import torch

from vector_quants.quantizers.utils import compute_dist, pack_one, unpack_one

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
    [128, 245, 768],
)
@pytest.mark.parametrize(
    "num_embed",
    [1024, 2048],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
def test_dist(shape, embed_dim, num_embed, dtype):
    print(shape)
    x = torch.randn(shape + [embed_dim], dtype=dtype).cuda()
    y = torch.randn(num_embed, embed_dim, dtype=dtype).cuda()

    x_flatten, ps = pack_one(x, "* d")
    dist = compute_dist(x_flatten, y)
    indices = torch.argmin(dist, dim=-1)
    indices = unpack_one(indices, ps, "*")

    # groud truth
    x_flatten_ref = x.reshape(-1, embed_dim)
    dist_ref = (
        x_flatten_ref.pow(2).sum(1, keepdim=True)
        + y.pow(2).sum(1, keepdim=True).t()
        - 2 * x_flatten_ref @ y.t()
    )
    indices_ref = torch.argmin(dist_ref, dim=-1)
    indices_ref = indices_ref.view(shape)

    torch.testing.assert_close(dist_ref, dist)
    torch.testing.assert_close(indices_ref, indices)
