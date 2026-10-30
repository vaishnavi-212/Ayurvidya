"""Citation checker: does the source a sentence cites actually contain what the sentence says?

For every answer sentence with tags like [R2] or [S4, G1] we compare its content words with the words of
the cited texts (script-folded, crude 6-letter stems). English sentences are only compared with sources
that contain English (the saved steps); against a Devanagari-only verse they are 'unchecked', not 'weak',
because word overlap cannot judge a translation. Cheap, no LLM call, and good enough to flag sentences a
student should double-check. Reported per answer as citation precision in the evaluation.
"""
from __future__ import annotations

import re

from .query import content_words, soft_overlap

TAGS = re.compile(r"\[((?:[RSGN]\d+)(?:\s*,\s*[RSGN]?\d+)*)\]")
SANSKRIT = re.compile(r"[ऀ-ॿ]+|\b\w*[āīūṛṝḷṃḥṅñṭḍṇśṣ]\w*\b")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।॥])\s+|\n+", text) if s.strip()]


def parse_tags(sentence: str) -> list[str]:
    out = []
    for group in TAGS.findall(sentence):
        letter = "R"
        for piece in re.split(r"\s*,\s*", group):
            if piece[:1].isalpha():
                letter = piece[0]
                out.append(piece)
            else:
                out.append(letter + piece)
    return out


def check_citations(answer: str, tag_texts: dict[str, str], min_support: float = 0.25) -> list[dict]:
    results = []
    for sent in split_sentences(answer):
        tags = [t for t in parse_tags(sent) if t in tag_texts]
        if not tags:
            continue
        body = TAGS.sub(" ", sent)
        cited = " ".join(tag_texts[t] for t in tags)
        if re.search(r"[A-Za-z]{3}", cited):
            words = content_words(body)
        else:                                       # Devanagari-only source: compare Sanskrit words only
            words = content_words(" ".join(SANSKRIT.findall(body)))
        if len(words) < 2:
            status, score = "unchecked", None
        else:
            score = round(soft_overlap(words, cited) / len(words), 3)
            status = "supported" if score >= min_support else "weak"
        results.append({"sentence": sent, "tags": tags, "status": status, "score": score})
    return results


def citation_precision(checks: list[dict]) -> float | None:
    judged = [c for c in checks if c["status"] != "unchecked"]
    return round(sum(c["status"] == "supported" for c in judged) / len(judged), 3) if judged else None