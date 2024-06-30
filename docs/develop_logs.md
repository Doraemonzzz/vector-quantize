# Develop logs

Here is a list of the development logs.

## Here are some summarized experiences:

1. Group/Residual Vq is the most useful;
2. Entropy loss: $\mathbb{E}[H(p(\mathbf{z}))] - H[\mathbb{E}(p(\mathbf{z}))]$:
   1. For each sample, minimizing entropy ensures determinism (the first term);
   2. For each class, maximizing entropy indicates a uniform distribution (the second term);
   3. It is very useful for Lfq but not very effective for Vq.

## Here is a list of the features that need to be added.

Indice has some bug.
gumbel debug:
```
-dist as logit not converge

conv2d(input) as logit converge.

use linear(input) as logit.

kl loss(need to use -?)
```

Need check:
```
entropy loss(rerun fsq, vq), old version use dist as logits, test -dist.
kl loss(need to use -?)
```

Need clear:
```
vector_quants/quantizers/utils.py
gumbel_vector_quantizer
```

Quant Method:
- [x] Vq.
- [x] Gvq.
- [x] Hvq.
- [x] Cvq.
- [x] Rvq.
- [x] Fsq.
- [x] Lfq.

Methods for finding the codebook.
- [x] Kmeans.
- [x] Ema.
- [ ] Gumbel.

Loss:
- [x] L1 loss.
- [x] L2 loss.
- [x] Perceptual loss.
- [ ] KL loss.
- [x] Entropy loss.
- [ ] Gan loss.
- [ ] Orthogonal loss.

Backbone.
- [ ] Convnet.
- [ ] Transformer.
- [ ] Linear Transformer.

Tradeoff between resolution and number of features.
- [ ] f8, f32.
- [ ] Multi scale like var.

Others:
- [ ] L2 normalize.

## 240624
Todo:
Quant Method:
- [x] Rvq.
- [x] Fsq.
- [x] Lfq.
- [x] Raq.

Methods for finding the codebook:
- [ ] Gumbel.

Loss:
- [x] Entropy loss.

## 240701
Methods for finding the codebook:
- [ ] Gumbel.
- [ ] Training softmax, inference onehot.

Loss:
- [ ] Gan loss.
- [ ] KL loss.

Add ar:
- [ ] Model.
- [ ] Training.

Others:
- [ ] L2 norm.
