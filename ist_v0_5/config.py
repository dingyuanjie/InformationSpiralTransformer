"""Configuration for IST v0.5 Level A."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class V05Config:
    vocab_size: int = 128
    hidden_size: int = 64
    heads: int = 4
    layers: int = 2
    chunk_size: int = 16
    evidence_capacity: int = 12
    evidence_span: int = 4
    writes_per_chunk: int = 4
    reads_per_query: int = 3
    core_slots: int = 4
    dropout: float = 0.0
    age_decay: float = 0.02
    usage_bonus: float = 0.1
    novelty_weight: float = 0.2
    redundancy_weight: float = 0.1
    evidence_gate_init: float = 0.0
    core_gate_init: float = 0.0
    reader_temperature: float = 1.0
    reranker_weight: float = 0.0

    def validate(self) -> None:
        integer_fields = ("vocab_size", "hidden_size", "heads", "layers", "chunk_size",
                          "evidence_capacity", "evidence_span", "writes_per_chunk",
                          "reads_per_query", "core_slots")
        if any(getattr(self, key) < 1 for key in integer_fields):
            raise ValueError("all size fields must be positive")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.writes_per_chunk > self.chunk_size - self.evidence_span + 1:
            raise ValueError("writes_per_chunk exceeds available evidence windows")
        if self.reads_per_query > self.evidence_capacity:
            raise ValueError("reads_per_query exceeds evidence capacity")
        if self.reader_temperature <= 0:
            raise ValueError("reader_temperature must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "V05Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = cls.__dataclass_fields__
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        config.validate()
        return config

    def to_dict(self):
        return asdict(self)
