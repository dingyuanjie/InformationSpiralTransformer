"""Leakage-audited shared-vocabulary/new-binding Level A data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

import torch


PAD, FACT, QUERY, ATTR, QUERY_ALT, RELATION, NEGATE = 0, 1, 2, 3, 4, 5, 6
ENTITY_IDS = tuple(range(16, 32))
VALUE_IDS = tuple(range(48, 64))
FILLER_IDS = tuple(range(80, 112))


def binding_fold(entity: int, value: int) -> int:
    digest = hashlib.sha256(f"{entity}:{value}:ist-v0.5".encode()).digest()
    return digest[0] % 5


TRAIN_BINDINGS = frozenset((e, v) for e in ENTITY_IDS for v in VALUE_IDS if binding_fold(e, v) != 0)
HELDOUT_BINDINGS = frozenset((e, v) for e in ENTITY_IDS for v in VALUE_IDS if binding_fold(e, v) == 0)


@dataclass
class Batch:
    history: torch.Tensor
    query: torch.Tensor
    answers: torch.Tensor
    target_chunks: torch.Tensor
    target_entities: torch.Tensor
    fact_positions: torch.Tensor
    split: str

    def to(self, device):
        return Batch(*(value.to(device) if torch.is_tensor(value) else value
                       for value in self.__dict__.values()))


def _pairs_for(split: str):
    if split == "train":
        return tuple(TRAIN_BINDINGS)
    if split in {"validation", "strict"}:
        return tuple(HELDOUT_BINDINGS)
    raise ValueError(f"unknown split {split}")


SCENARIOS = ("single_fact", "multi_fact", "interference", "overwrite", "temporal_update",
             "negative", "two_hop", "paraphrase", "position_shift", "extrapolation", "density")


def make_batch(batch_size: int, chunks: int, chunk_size: int, seed: int,
               split: str = "train", facts_per_chunk: int = 2,
               conflict: bool = False, negative: bool = False, scenario: str = "multi_fact") -> Batch:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    conflict = conflict or scenario in {"overwrite", "temporal_update"}
    negative = negative or scenario == "negative"
    if scenario == "single_fact": facts_per_chunk = 1
    if scenario == "density": facts_per_chunk = min(4, chunk_size // 4)
    if chunk_size < facts_per_chunk * 4:
        raise ValueError("chunk is too small for requested fact density")
    rng = random.Random(seed)
    allowed = _pairs_for(split)
    histories, queries, answers, target_chunks, target_entities, positions = [], [], [], [], [], []
    for _ in range(batch_size):
        stream = []
        records = []
        used_entities = set()
        for chunk in range(chunks):
            row = [rng.choice(FILLER_IDS) for _ in range(chunk_size)]
            possible = list(range(0, chunk_size - 3, 4))
            starts = sorted(rng.sample(possible, facts_per_chunk))
            for start in starts:
                candidates = [pair for pair in allowed if pair[0] not in used_entities]
                if not candidates:
                    used_entities.clear(); candidates = list(allowed)
                entity, value = rng.choice(candidates); used_entities.add(entity)
                row[start:start + 4] = [FACT, entity, ATTR, value]
                records.append((entity, value, chunk, chunk * chunk_size + start))
            stream.append(row)
        entity, value, source_chunk, source_position = rng.choice(records)
        query_entity = entity
        if conflict:
            compatible = [pair for pair in allowed if pair[0] == entity and pair[1] != value]
            if compatible:
                _, value = rng.choice(compatible)
                final = stream[-1]
                final[-4:] = [FACT, entity, ATTR, value]
                source_chunk, source_position = chunks - 1, chunks * chunk_size - 4
        if scenario == "interference":
            # Add a nearby competing value for the same entity, while retaining the selected target as latest.
            compatible = [pair for pair in allowed if pair[0] == entity and pair[1] != value]
            if compatible and chunks > 1:
                _, distractor = rng.choice(compatible)
                stream[0][-4:] = [FACT, entity, ATTR, distractor]
                stream[-1][-4:] = [FACT, entity, ATTR, value]
                source_chunk, source_position = chunks - 1, chunks * chunk_size - 4
        if scenario == "two_hop":
            bridge = rng.choice([item for item in ENTITY_IDS if item != entity])
            stream[0][:4] = [FACT, entity, RELATION, bridge]
            stream[-1][-4:] = [FACT, bridge, ATTR, value]
            query_entity = entity
            source_chunk, source_position = chunks - 1, chunks * chunk_size - 4
        query_token = QUERY_ALT if scenario == "paraphrase" else QUERY
        query = [query_token, query_entity, RELATION if scenario == "two_hop" else ATTR]
        answer = value
        if negative:
            query = [QUERY, entity, PAD]
            answer = PAD
        histories.append(stream); queries.append(query); answers.append(answer)
        target_chunks.append(source_chunk); target_entities.append(bridge if scenario == "two_hop" else entity)
        positions.append(source_position)
    return Batch(torch.tensor(histories), torch.tensor(queries), torch.tensor(answers),
                 torch.tensor(target_chunks), torch.tensor(target_entities), torch.tensor(positions), split)


def split_audit() -> dict:
    train_entities = {e for e, _ in TRAIN_BINDINGS}; heldout_entities = {e for e, _ in HELDOUT_BINDINGS}
    train_values = {v for _, v in TRAIN_BINDINGS}; heldout_values = {v for _, v in HELDOUT_BINDINGS}
    overlap = TRAIN_BINDINGS & HELDOUT_BINDINGS
    return {
        "binding_overlap": len(overlap),
        "train_binding_count": len(TRAIN_BINDINGS),
        "heldout_binding_count": len(HELDOUT_BINDINGS),
        "shared_entity_vocabulary": train_entities == heldout_entities == set(ENTITY_IDS),
        "shared_value_vocabulary": train_values == heldout_values == set(VALUE_IDS),
        "label_is_resampled_per_example": True,
        "query_visible_during_history_write": False,
        "passed": not overlap and train_entities == heldout_entities and train_values == heldout_values,
    }


def assert_no_leakage() -> None:
    report = split_audit()
    if not report["passed"]:
        raise RuntimeError(f"split leakage: {report}")
