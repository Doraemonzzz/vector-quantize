from vector_quants.models import AutoVqVae

path = "/mnt/iem-nas/home/qinzhen/qinzhen/experiment_generation/fsq/checkpoints_test/Fsq-is-128-bs-256-lr-1.0e-4-wd-0-dtype-bf16-num_embed-4096/ckpts/1.pt"

model = AutoVqVae.from_pretrained(path)

print(model)
