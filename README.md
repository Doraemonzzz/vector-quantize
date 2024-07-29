

# Todo
For the complete development logs, please refer to [here](./docs/develop_logs.md).

- [x] Add simple fsq. (use [0, 1] instead of [-1, 1], which makes the code more simpler.)
- [x] Mix presion training.
- [x] DDP evaluation.
- [x] Add wandb logger.
- [ ] Add large batch size result.
- [x] Add resume code.
- [ ] Check ema and lfq bug.
- [ ] Test Transformer arch.
- [ ] Test Linear Attention arch.
- [ ] Perceptual loss ablation.
- [ ] Update fid evaluation.
- [ ] Add gradient accumulation. (rerun this)
- [x] Rerun bf16 results.
- [x] Add trainer.
- [x] Add metrics.
- [x] Update eval code, ref https://github.dev/SerezD/vqvae-vqgan-pytorch-lightning
- [x] Update code book usage.
- [x] Use dataclass and YAML to manage config.
- [x] Update eval
- [x] Update resume, add yaml file
- [ ] Residual vq bug.
- [ ] Ema gradient acc.
- [x] Ema codebook loss bug.
- [ ] Update evaluation bug.
- [ ] Gumbel bin count bug.
- [x] Update residual in rvq.

Metric check list.
- [ ] fid.
- [x] codebook usage.
- [x] adversarial loss.
- [x] is.
- [x] l1.
- [x] l2.
- [ ] mse.
- [x] lpips.
- [x] perceptual loss.
- [ ] psnr. (need check)
- [x] ssim.
- [x] gnorm.


Algorithm check list.
- [ ] vqvae.
- [ ] vqvae ema.
- [ ] vqvae gumbel.
- [ ] vqvae.
- [ ] fsq.

Loss check list.
- [ ] L1, L2 loss?
- [ ] Gumbel, kl loss.
- [ ] Entropy loss in maskgit.
- [ ] Perceptual loss.
- [ ] Gan loss.


# Results

## Repreduced result

Using the default config in [here](https://github.com/duchenzhuang/FSQ-pytorch). Some main config are list as follows:

```
dtype: fp32
batch size: 32 * 8
lr: 1e-4
```


| name | levels      | fid       | l1_loss | precep_loss | codebook_usage |
|------|-------------|-----------|---------|-------------|----------------|
| ema  | -           | 88.6831   | 0.3528  | 0.3592      | 0.0006         |
| lfq  | -           | 229.3754  | 0.6472  | 0.5462      | 0.0002         |
| fsq  | 8-6-5       | 67.5584   | 0.2772  | 0.3227      | 1.0000         |
| fsq  | 8-5-5-5     | 58.5167   | 0.2466  | 0.2850      | 1.0000         |
| fsq  | 7-5-5-5-5   | 37.6831   | 0.2136  | 0.2310      | 1.0000         |
| fsq  | 8-8-8-6-5   | 28.3809   | 0.1917  | 0.1933      | 1.0000         |
| fsq  | 8-8-8-5-5-5 | 24.8071   | 0.1812  | 0.1758      | 0.9987         |
| sfsq | 8-8-8-5-5-5 | 25.5078   | 0.1834  | 0.1796      | 0.9992         |

The result of ema and lfq seems werid, need check this.

## Bf16 result

Config:
```
dtype: bf16
batch size: 32 * 8
lr: 1e-4
```

| name | levels      | fid       | l1_loss | precep_loss | codebook_usage |
|------|-------------|-----------|---------|-------------|----------------|
| ema  | -           | 151.7567  | 0.4831  | 0.4192      | 0.0911         |
| lfq  | -           | 235.0212  | 0.6491  | 0.5473      | 0.0286         |
| fsq  | 8-6-5       | 65.4644   | 0.2645  | 0.3142      | 1.0000         |
| fsq  | 8-5-5-5     | 48.4387   | 0.2480  | 0.2699      | 1.0000         |
| fsq  | 7-5-5-5-5   | 37.9094   | 0.2175  | 0.2323      | 1.0000         |
| fsq  | 8-8-8-6-5   | 30.7434   | 0.1959  | 0.1998      | 1.0000         |
| fsq  | 8-8-8-5-5-5 | 25.4654   | 0.1824  | 0.1799      | 0.9985         |
| sfsq | 8-8-8-5-5-5 | 25.5291   | 0.1843  | 0.1812      | 0.9991         |

It seems that there are no major issues with mixed-precision training under fsq, but we need to check the results for lfq and ema.

## Notes

1. The main reason for Fsq work is hierarchical decomposition?
   1. Kmeans -> Hierarchical Kmeans, e.g, (n * m, d) -> (n + m, d)
   2. Check fsq-1: e.g, x -> [0, 1023]
   3. Map the input to a one-dimensional number in the range [0, M - 1], and then perform radix decomposition.
   4.
2. It is important to convert it into an integer, rather than its corresponding embedding?


## Acknowledgement

1. Our code is adapt from [FSQ-pytorch](https://github.com/duchenzhuang/FSQ-pytorch), and we would like to thank these teams for their selfless sharing.
2. https://github.com/karpathy/deep-vector-quantization for origin VQVAE.
3. https://github.com/zh460045050/VQGAN-LC/
4. https://github.com/kakaobrain/rq-vae-transformer/
5. https://github.com/SerezD/vqvae-vqgan-pytorch-lightning
6. https://github.com/dome272/VQGAN-pytorch


## Reference
vqvae gumbel
1. https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/gumbel_vector_quantizer.py
2. https://arxiv.org/abs/1910.05453
3. https://github.com/kakaobrain/rq-vae-transformer/
4. https://gist.github.com/sekstini/7f089f71d4b975ec8bde37d878b514d0
5. Vector quantization loss analysis in VQGANs: a single-GPU ablation study for image-to-image synthesis
6. Hierarchical Residual Learning Based Vector Quantized Variational Autoencoder for Image Reconstruction and Generation
7. https://github.com/luv91/VQGAN_Project/
8. https://github.com/S-aiueo32/lpips-pytorch/
9. https://github.com/pytorch/torchtitan
10. https://github.com/GoGoDuck912/pytorch-vector-quantization/
11. https://github.com/archinetai/bitcodes-pytorch/
12. https://github.com/guglielmocamporese/deep-vector-quantization
13. https://minyoungg.github.io/vqtorch/
14. https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html
15. https://dsp.stackexchange.com/questions/27551/8-times-8-block-matrix-in-jpeg-image-compression
16. https://stackoverflow.com/questions/10780425/why-jpeg-compression-processes-image-by-8x8-blocks

## Citation
If you find this repo useful, please consider citing my work:
```
@software{qin2024vq,
  title  = {Vector quantize: An open-source library for ablating various VQ methods.},
  author = {Zhen Qin},
  url    = {https://github.com/Doraemonzzz/vector-quantize},
  month  = {7},
  year   = {2024}
}
```
