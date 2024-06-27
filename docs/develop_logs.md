# Develop logs

Here is a list of the development logs.

## Here is a list of the features that need to be added.

Indice has some bug.

Raq debug:
```
diff = 0, number = round_ste(F.sigmoid(latent) * d), code = number / d => ok
diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x), number = round_ste(F.sigmoid(latent) * d), code = number / d => no ok
diff = 0, number = round_ste(F.sigmoid(latent) * d), code = torch.sin(torch.pi * (number / d - 0.5)) => ok
```

Lfq debug:
```
diff = self.commitment_loss_weight * F.mse_loss(x_quant.detach(), x), x_quant = x + (x_quant - x).detach() => fid太高
+l2 norm
```

Entropy loss:
Jenson不等式：
$$
\mathcal{L}_{\text {entropy }}=\mathbb{E}[H(q(\mathbf{z}))]-H[\mathbb{E}(q(\mathbf{z}))]\ge 0
$$
等号成立当且仅当$q(z)=\frac 1 n$。

Quant Method:
- [x] Vq.
- [x] Gvq.
- [x] Hvq.
- [x] Cvq.
- [ ] Rvq.
- [ ] Fsq.
- [ ] Lfq.

Methods for finding the codebook.
- [x] Kmeans.
- [x] Ema.
- [ ] Gumbel.

Loss:
- [ ] L1 loss.
- [ ] L2 loss.
- [ ] Perceptual loss.
- [ ] KL loss.
- [ ] Entropy loss.
- [ ] Gan loss.

Backbone.
- [ ] Convnet.
- [ ] Transformer.
- [ ] Linear Transformer.

Tradeoff between resolution and number of features.
- [ ] f8, f32.
- [ ] Multi scale like var.


## 240624
Todo:
Quant Method:
- [ ] Rvq.
- [ ] Fsq.
- [ ] Lfq.

Loss:
- [ ] Gumbel
