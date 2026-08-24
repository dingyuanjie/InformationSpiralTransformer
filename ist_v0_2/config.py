"""Configuration objects for the minimal IST v0.2 hierarchical Memory prototype."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FastMemoryConfig:
    enabled: bool = True
    slots: int = 32


@dataclass
class SlowMemoryConfig:
    enabled: bool = True
    slots: int = 8
    update_interval: int = 1
    retention_bias: float = 2.0


@dataclass
class EpisodicMemoryConfig:
    enabled: bool = True
    slots: int = 64
    top_k: int = 4
    age_weight: float = 1.0
    usage_weight: float = 1.0
    importance_weight: float = 1.0
    redundancy_weight: float = 0.25


@dataclass
class RouterConfig:
    enabled: bool = True
    mode: str = "soft"
    temperature: float = 1.0


@dataclass
class ConsolidationConfig:
    enabled: bool = True


@dataclass
class HierarchicalMemoryConfig:
    enabled: bool = True
    fast: FastMemoryConfig = field(default_factory=FastMemoryConfig)
    slow: SlowMemoryConfig = field(default_factory=SlowMemoryConfig)
    episodic: EpisodicMemoryConfig = field(default_factory=EpisodicMemoryConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)

    def validate(self, hidden_size: int) -> None:
        if self.router.mode not in {"soft", "hard_straight_through", "disabled"}:
            raise ValueError("router mode must be soft, hard_straight_through, or disabled")
        for name, slots in (("fast", self.fast.slots), ("slow", self.slow.slots),
                            ("episodic", self.episodic.slots)):
            if slots < 1:
                raise ValueError(f"{name} slots must be positive")
        if not 1 <= self.episodic.top_k <= self.episodic.slots:
            raise ValueError("episodic top_k must be between 1 and episodic slots")
        if hidden_size % 8:
            raise ValueError("hidden size must be divisible by 8")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None):
        if isinstance(value, cls):
            return value
        # Support the same dataclass loaded through the package-qualified name
        # and through the script-compatible top-level import name.
        if value is not None and hasattr(value, "to_dict"):
            value = value.to_dict()
        value = value or {}
        return cls(
            enabled=value.get("enabled", True),
            fast=FastMemoryConfig(**value.get("fast", {})),
            slow=SlowMemoryConfig(**value.get("slow", {})),
            episodic=EpisodicMemoryConfig(**value.get("episodic", {})),
            router=RouterConfig(**value.get("router", {})),
            consolidation=ConsolidationConfig(**value.get("consolidation", {})),
        )
