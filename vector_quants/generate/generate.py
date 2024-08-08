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

    for k in range(steps):
        cond_idx = c if k == 0 else None
        logits, past_key_values = model(
            idx=idx, cond_idx=cond_idx, past_key_values=past_key_values
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
