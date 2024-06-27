from vector_quants.models import AutoVqVae

path = ""

model = AutoVqVae.from_pretrained(path)

print(model)
