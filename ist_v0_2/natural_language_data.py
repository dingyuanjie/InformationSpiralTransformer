"""Deterministic, shortcut-audited natural-language bridge data.

UTF-8 bytes avoid a fitted vocabulary and keep train/held-out/OOD entity names
truly unseen. Answers are shuffled multiple-choice indices represented by 16
reserved output ids, so scoring is exact without substring heuristics.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict


PAD_ID = 256
ANSWER_BASE = 257
ANSWER_CLASSES = 16
VOCAB_SIZE = ANSWER_BASE + ANSWER_CLASSES

NAMES = {
    "train": ("Alice", "Ben", "Clara", "Daniel", "Elena", "Farah"),
    "validation": ("Grace", "Hector", "Iris", "Jonah"),
    "held_out": ("Keira", "Liam", "Mina", "Noah", "Priya", "Rafael"),
    "ood": ("Aiko", "Bartosz", "Chiamaka", "Dmitri", "Eszter", "Femi"),
}
OBJECTS = {
    "train": ("red key", "passport", "silver watch", "notebook", "camera"),
    "validation": ("blue folder", "museum ticket", "brass key"),
    "held_out": ("violet badge", "train pass", "glass pendant", "field journal"),
    "ood": ("ceramic token", "encrypted drive", "meteorite sample", "violin bow"),
}
LOCATIONS = {
    "train": ("the second drawer in the study", "the top shelf of the bedroom wardrobe",
              "the locked cabinet in the kitchen", "the wooden box in the garage"),
    "validation": ("the archive room's lower cupboard", "the basket beside the balcony door"),
    "held_out": ("the narrow locker behind the workshop", "the blue case beneath the guest bed",
                 "the third compartment of the library desk"),
    "ood": ("bay seven of the orbital storeroom", "the climate vault under the west greenhouse",
            "the sealed drawer aboard the research vessel"),
}
FACT_TEMPLATES = {
    "train": ("{name} placed the {obj} in {location}.", "Before leaving, {name} stored the {obj} in {location}."),
    "validation": ("{name} carefully left the {obj} inside {location}.",),
    "held_out": ("For safekeeping, the {obj} was put by {name} in {location}.",
                 "The place {name} chose for the {obj} was {location}."),
    "ood": ("After checking the inventory, {name} secured the {obj} within {location}.",),
}
QUERY_TEMPLATES = {
    "train": ("Where did {name} put the {obj}?", "Where is {name}'s {obj} stored?"),
    "validation": ("Which location contains the {obj} that belongs to {name}?",),
    "held_out": ("Recall the storage place selected by {name} for the {obj}.",
                 "In which of these places would {name} find the {obj}?"),
    "ood": ("According to the earlier account, identify where {name} secured the {obj}.",),
}
DISTRACTORS = (
    "The committee reviewed the weekly timetable and postponed two routine meetings.",
    "Outside, a light rain crossed the courtyard while the maintenance crew checked the windows.",
    "A local newspaper described a community garden that opened near the railway station.",
    "During lunch, several colleagues discussed a documentary about coastal wildlife.",
    "The library received new maps, repaired three lamps, and updated its visitor notice.",
    "A delivery driver confirmed the afternoon route before returning an empty container.",
    "The weather report predicted cooler evenings and a clear sky later in the week.",
    "Someone watered the plants, sorted the mail, and wrote tomorrow's appointments on a calendar.",
)


@dataclass
class NLExample:
    chunks: list[list[int]]
    target: int
    metadata: dict

    def to_dict(self): return asdict(self)


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(x for x in ids if 0 <= x < 256).decode("utf-8", errors="replace")


def _pool(table, split):
    return table[split]


def generate_nl1(seed: int, split="train", distance=2048, chunk_size=512,
                 option_count=8) -> NLExample:
    if split not in NAMES: raise ValueError(f"unknown split: {split}")
    if distance < chunk_size or distance % chunk_size: raise ValueError("distance must be >=1 chunk and divisible by chunk_size")
    if not 2 <= option_count <= 8: raise ValueError("option_count must be between 2 and 8")
    rng = random.Random(seed)
    name=rng.choice(_pool(NAMES,split)); obj=rng.choice(_pool(OBJECTS,split))
    location=rng.choice(_pool(LOCATIONS,split))
    fact=rng.choice(_pool(FACT_TEMPLATES,split)).format(name=name,obj=obj,location=location)
    alternatives=list(dict.fromkeys(sum((list(v) for v in LOCATIONS.values()), [])))
    alternatives.remove(location); options=rng.sample(alternatives, option_count-1)+[location]
    rng.shuffle(options); answer_index=options.index(location)
    labels="ABCDEFGHJKLMNOPQ"[:len(options)]
    query=rng.choice(_pool(QUERY_TEMPLATES,split)).format(name=name,obj=obj)
    query += "\nChoices: " + " ".join(f"{labels[i]}) {value}" for i,value in enumerate(options)) + "\nAnswer:"
    total_chunks=distance//chunk_size
    chunks=[]
    for chunk_index in range(total_chunks):
        text=""
        while len(encode(text)) < chunk_size * 2:
            sentence=rng.choice(DISTRACTORS)
            if split=="ood": sentence=sentence.replace("The ","In an unrelated report, the ",1)
            text += sentence+" "
        raw=encode(text)
        reserved = len(encode(query)) + 8 if total_chunks == 1 else 8
        upper = chunk_size-len(encode(fact))-reserved
        if chunk_index == 0 and upper <= 8:
            raise ValueError("fact and query do not fit; reduce option_count or increase chunk_size")
        offset=rng.randrange(8,upper) if chunk_index==0 else None
        if chunk_index==0: raw[offset:offset+len(encode(fact))]=encode(fact)
        if chunk_index==total_chunks-1: raw[-len(encode(query)):]=encode(query)
        chunks.append((raw[:chunk_size]+[PAD_ID]*chunk_size)[:chunk_size])
    metadata={"seed":seed,"split":split,"distance":distance,"needle_chunk":0,"needle_byte_offset":offset,
              "name":name,"object":obj,"location":location,"fact":fact,"query":query,
              "options":options,"answer_index":answer_index,
              "shortcut_controls":["template_shuffle","entity_replacement","position_shift","query_rewrite"]}
    return NLExample(chunks,ANSWER_BASE+answer_index,metadata)


def audit_example(example: NLExample) -> dict:
    m=example.metadata; last=decode(example.chunks[-1])
    fact_in_last=m["fact"] in last
    expected_same_chunk = m["distance"] == len(example.chunks[-1])
    return {"passed":(expected_same_chunk or not fact_in_last) and m["needle_byte_offset"] != 0 and m["location"] in m["options"],
            "fact_absent_from_final_chunk":not fact_in_last,"needle_position_shifted":m["needle_byte_offset"]!=0,
            "same_chunk_calibration":expected_same_chunk,
            "answer_options_shuffled":True, "correct_option_index":m["answer_index"]}
