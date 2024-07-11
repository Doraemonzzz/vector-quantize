# Develop logs

Here is a list of the development logs.

## Here are some summarized experiences:

1. Group/Residual Vq is the most useful;
2. Entropy loss: $\mathbb{E}[H(p(\mathbf{z}))] - H[\mathbb{E}(p(\mathbf{z}))]$:
   1. For each sample, minimizing entropy ensures determinism (the first term);
   2. For each class, maximizing entropy indicates a uniform distribution (the second term);
   3. It is very useful for Lfq but not very effective for Vq.
3. During training, Gumbel VQ needs to use hard=True to yield reasonable results. There was a bug version run in between that performed normalization on the "b c h w" dimension's w axis with hard=False, and it still produced relatively reasonable results, with an initial FID of around 5. However, as training progressed, the FID increased to 20.
4. Softmax VQ doesn't work at all. The current hypothesis is that it is best to use the one-hot form during the training phase; otherwise, the training and inference phases will be inconsistent.

## Notes
FreqPatchEmbed:
```
    # # v1
    # def forward(self, x):
    #     y = rearrange(
    #         x,
    #         "b c (p1 h) (p2 w) -> b (p1 p2 c) h w",
    #         h=self.num_h_patch,
    #         w=self.num_w_patch,
    #     )
    #     y = dct_2d(y, norm="ortho")
    #     y = rearrange(y, "b d h w -> b (h w) d")[:, self.indices]
    #     y = self.to_patch_embedding(y)

    #     return y

    # # v2
    # def forward(self, x):
    #     y = rearrange(
    #         x,
    #         "b c h w -> b c (h w)",
    #     )[:, :, self.indices]

    #     y = rearrange(y, "b c (n d) -> b n (d c)", n=self.num_patch)
    #     y = self.to_patch_embedding(y)

    #     return y

    def forward(self, x):
        y = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            h=self.num_h_patch,
            w=self.num_w_patch,
        )
        y = self.to_patch_embedding(y)

        return y
```

FreqTransformer:
```
        # # v1
        # # (b c h w)
        # x = self.patch_embed(x)
        # x = rearrange(dct_2d(x, norm="ortho"), "b c h w -> b (h w) c")
        # x = self.final_norm(x)
        # # convert to zigzag order
        # x = x[:, self.indices, :]

        # # v2
        # # (b c h w)
        # x = self.patch_embed(dct_2d(x))
        # x = rearrange(x, "b c h w -> b (h w) c")

        # # convert to zigzag order
        # x = x[:, self.indices, :]
```

## Here is a list of the features that need to be added.

Indice has some bug.
gumbel:
```
kl loss(need to use -?)
ce loss
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
ModelConfig
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
- [ ] Align interface of patch embedding.

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
- [x] Gumbel.
- [x] Training softmax, inference onehot.
  - [x] Not work.

Loss:
- [ ] Gan loss.
- [x] KL loss.
  - [ ] Add summary

Add ar:
- [ ] Model.
- [ ] Training.

Others:
- [x] L2 norm.
  - [x] No use.


## 240707

- [ ] Test freq ar.
- [ ] Test vit + extra token.
- [ ] Test rcq.
- [ ] Add Gan loss.
