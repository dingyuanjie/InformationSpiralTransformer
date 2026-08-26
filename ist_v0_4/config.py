"""Configuration for IST v0.4 cognitive event memory."""
from dataclasses import dataclass


@dataclass
class CognitiveMemoryConfig:
    event_span: int = 24
    event_stride: int = 8
    working_events: int = 4
    episodic_events: int = 16
    semantic_slots: int = 8
    admissions_per_chunk: int = 2
    retrieved_events: int = 3
    novelty_weight: float = 1.0
    surprise_weight: float = 1.0
    age_decay: float = 0.04
    access_bonus: float = 0.25
    consolidation_accesses: int = 3
    semantic_mix: float = 0.1

    def validate(self, hidden_size: int) -> None:
        for name in ("event_span", "working_events", "episodic_events", "semantic_slots",
                     "admissions_per_chunk", "retrieved_events", "event_stride"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.admissions_per_chunk > self.episodic_events:
            raise ValueError("admissions_per_chunk exceeds episodic capacity")
        if self.event_stride > self.event_span:
            raise ValueError("event_stride cannot exceed event_span")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
