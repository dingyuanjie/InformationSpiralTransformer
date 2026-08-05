import random

import torch


class LengthCurriculum:
    """Select sequence lengths using fixed, random or staged curricula."""

    def __init__(self, mode="curriculum", minimum=32, maximum=512):
        self.mode = mode
        self.minimum = minimum
        self.maximum = maximum

    def sample(self, step, total_steps):
        if self.mode == "fixed":
            return random.choice([self.minimum, min(64, self.maximum)])
        if self.mode == "random":
            ceiling = self.maximum
        elif self.mode == "curriculum":
            progress = step / max(total_steps, 1)
            if progress < 0.20:
                ceiling = min(64, self.maximum)
            elif progress < 0.40:
                ceiling = min(128, self.maximum)
            elif progress < 0.70:
                ceiling = min(256, self.maximum)
            else:
                ceiling = self.maximum
        else:
            raise ValueError(f"unknown length mode: {self.mode}")
        # Multiples of 8 keep batches efficient and experiments reproducible.
        choices = range(self.minimum, ceiling + 1, 8)
        return random.choice(list(choices))


def make_random_needle_batch(batch_size, length, vocab_size, device):
    """Create [NEEDLE] target ... [QUERY] [MASK] retrieval examples."""
    if length < 5:
        raise ValueError("needle retrieval sequences must contain at least 5 tokens")
    mask_token = vocab_size
    needle_marker = vocab_size + 1
    query_marker = vocab_size + 2
    targets = torch.randint(vocab_size, (batch_size,), device=device)
    tokens = torch.randint(vocab_size, (batch_size, length), device=device)
    positions = torch.randint(0, length - 3, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    tokens[rows, positions] = needle_marker
    tokens[rows, positions + 1] = targets
    tokens[:, -2] = query_marker
    tokens[:, -1] = mask_token
    distances = length - 2 - (positions + 1)
    return tokens, targets, distances


def distance_bucket(distance):
    for upper in (64, 128, 256, 512, 1024):
        if distance <= upper:
            return f"{1 if upper == 64 else upper // 2 + 1}-{upper}"
    return "1025+"
