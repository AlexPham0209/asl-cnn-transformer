import torch

print(torch.linspace(start=0, end=10, steps=5, dtype=int))
print(torch.randint(low=0, high=180, size=(10,)).sort()[0])
