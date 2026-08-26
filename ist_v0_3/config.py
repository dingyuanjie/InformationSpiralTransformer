"""Configuration for IST v0.3 provenance-preserving token memory."""
from dataclasses import dataclass


@dataclass
class SourceTokenMemoryConfig:
    capacity: int = 32
    writes_per_chunk: int = 8
    reads_per_query: int = 4
    heads: int = 8
    injection_layer: int = -4
    initial_gate: float = -0.01

    def validate(self, hidden_size: int) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if not 1 <= self.writes_per_chunk <= self.capacity:
            raise ValueError("writes_per_chunk must be in [1, capacity]")
        if not 1 <= self.reads_per_query <= self.capacity:
            raise ValueError("reads_per_query must be in [1, capacity]")
        if hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")

