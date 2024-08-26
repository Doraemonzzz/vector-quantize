import torch
import torch.nn.functional as F
from einops import pack, unpack


def top_k_logits(logits, k):
    k = min(k, logits.shape[-1])
    # only sample topk
    v, ix = torch.topk(logits, k, dim=-1)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float("inf")
    return out


@torch.no_grad()
def sample(model, steps, c=None, idx=None, temperature=1.0, top_k=100):
    model.eval()
    shape = [steps]
    # prefill
    past_key_values = None

    if idx is not None:
        start = idx.shape[1]
        x = idx
    else:
        start = 0
        x = None

    for k in range(start, steps):
        cond_idx = c if k == 0 else None
        logits, past_key_values = model(
            idx=x, cond_idx=cond_idx, past_key_values=past_key_values, shape=shape
        )
        # get the last token's logits
        # b n g V -> b 1 g V
        logits = (
            logits[
                :,
                -1:,
            ]
            / temperature
        )

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        # probs: b n g v
        probs, ps = pack([probs], "* v")
        x = torch.multinomial(probs, num_samples=1)[:, 0]
        x = unpack(x, ps, "*")[0]

        idx = torch.cat([idx, x], dim=1) if k != 0 else x

    del past_key_values

    return idx


def test_sample_with_kv_cache(model, c, steps, temperature=1.0, top_k=100):
    model.eval()
    # with kv cache
    # prefill
    past_key_values = None
    x = None

    for k in range(steps):
        cond_idx = c if k == 0 else None
        logits, past_key_values = model(
            idx=x, cond_idx=cond_idx, past_key_values=past_key_values
        )
        # get the last token's logits
        logits = (
            logits[
                :,
                -1:,
            ]
            / temperature
        )

        x = torch.argmax(logits, dim=-1)

        idx = torch.cat([idx, x], dim=1) if k > 0 else x

    idx_kv_cache = idx.clone()

    # without kv cache
    # prefill
    past_key_values = None
    idx = None

    for k in range(steps):
        cond_idx = c if k == 0 else None

        logits, _ = model(idx=idx, cond_idx=cond_idx)
        x = torch.argmax(
            logits[
                :,
                -1:,
            ],
            dim=-1,
        )

        idx = torch.cat([idx, x], dim=1) if k > 0 else x

    print(torch.norm(idx.float() - idx_kv_cache).item())
    assert torch.norm(idx.float() - idx_kv_cache).item() == 0, "idx_kv_cache != idx"

    return idx
