

# Todo

- [x] Add simple fsq. (use [0, 1] instead of [-1, 1], which makes the code more simpler.)
- [x] Mix presion training.
- [x] DDP evaluation.
- [x] Add wandb logger.
- [ ] Add large batch size result.
- [ ] Check ema and lfq bug.
- [ ] Test Transformer arch.
- [ ] Test Linear Attention arch.

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
dtype: fp32
batch size: 32 * 8
lr: 1e-4
```

## Acknowledgement

Our code is adapt from [FSQ-pytorch](https://github.com/duchenzhuang/FSQ-pytorch), and we would like to thank these teams for their selfless sharing.
