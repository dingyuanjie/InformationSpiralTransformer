import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free sinusoidal positions that naturally extend in length."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.hidden_size)
        )
        encoding = torch.zeros(length, self.hidden_size, device=device)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)
