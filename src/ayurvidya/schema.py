"""Core data records shared by every module.

Keeping these as small dataclasses (not ORM models) means the same objects flow
through ingestion, indexing, the 7-step pipeline, the cache and the API.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# The 7 Ayurvidya steps, in order. Step numbers are used everywhere (cache keys, prompts, eval).
STEPS: dict[int, str] = {
    1: "sampradanam",
    2: "padavibhaga",
    3: "anvaya",
    4: "anvayartha",
    5: "bhavartha",
    6: "padakrityam",
    7: "dhvanitartha",
}

# Kinds of knowledge-base chunk. Each step retrieves only from the kinds it needs (see steps.py).
SOURCE_TYPES = (
    "mula",              # the shloka itself (NIIMH text)
    "commentary",        # Dalhana (Nibandhasangraha), Gayadasa (Nyayachandrika)
    "kosha",             # Amarakosha and other lexicon entries
    "dhatu",             # Dhatupatha / Dhatu Ratnavali root entries
    "grammar",           # vyakarana rules, sandhi/samasa notes, tantrayukti definitions
    "crosstext",         # parallel passages: Charaka, Ashtanga Hridaya, other Sushruta chapters
    "gold",              # scholar-validated 7-step annotations, used as few-shot examples
    "student_approved",  # student notes a teacher approved (scripts/review_notes.py)
)


@dataclass
class Shloka:
    """One retrievable unit of the mula text (a verse number, or a joined pair such as 1-2)."""
    id: str                     # e.g. "su.ni.1.3"
    samhita: str                # "sushruta"
    sthana: str                 # "nidana"
    chapter: int                # 1
    number: str                 # "3" or "1-2"
    text: str                   # Devanagari, lines joined with "\n"
    is_prose: bool = False      # gadya sutra (True) vs padya shloka (False)
    variant: str = ""           # bracketed pathantara printed by NIIMH, kept separately
    topic: str = ""             # prakarana label, if known

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A unit stored in the vector + keyword index."""
    id: str
    text: str
    source_type: str
    work: str = ""                         # e.g. "Nibandhasangraha", "Amarakosha"
    shloka_ids: list[str] = field(default_factory=list)  # verses this chunk is about
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Chunk":
        return Chunk(**d)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class StepResult:
    shloka_id: str
    step: int
    output: str
    valid: bool
    issues: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    attempts: int = 1
