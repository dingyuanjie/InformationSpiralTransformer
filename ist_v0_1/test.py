import torch

from model import InformationSpiralTransformer


model = InformationSpiralTransformer(
    vocab_size=5000,
    hidden_size=256,
    layers=4,
)

tokens = torch.randint(0, 5000, (2, 128))
output = model(tokens)

print(output.shape)
