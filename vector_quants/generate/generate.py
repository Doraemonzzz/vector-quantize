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
def sample(model, c, steps, temperature=1.0, top_k=100):
    model.eval()
    # prefill
    past_key_values = None
    idx = None
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

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        # probs: b n g v
        probs, ps = pack([probs], "* v")
        x = torch.multinomial(probs, num_samples=1)[:, 0]
        x = unpack(x, ps, "*")[0]

        idx = torch.cat([idx, x], dim=1) if k > 0 else x

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

    assert torch.norm(idx.float() - idx_kv_cache).item() == 0, "idx_kv_cache != idx"

    return idx
