"""Query understanding: what script the question is in, which verse it names, and which Sanskrit
words it is really about.

English questions ("Which Vayu moves in the mouth?") share no letters with the Devanagari verses, so
BM25 cannot match them. The glossary in data/synonyms.csv bridges that gap: 'mouth' adds वक्त्र मुख आस्य,
'Vayu' adds वात अनिल मारुत ... Sanskrit words typed in any script are matched by their folded prefix,
so inflected forms (वायुः, प्राणो) still hit. Edit the CSV to grow the glossary; no code change needed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .text import fold_for_search

GENERIC = {"vata", "disease", "cause", "sign", "seat", "action", "vitiated", "five"}
VERSE_RE = re.compile(r"(?:verse|shloka|sloka|śloka|श्लोक[ःा]?)\s*(?:no\.?\s*)?([0-9०-९]+)", re.I)


@dataclass
class QueryInfo:
    text: str
    script: str                          # devanagari | iast | english | mixed
    verse: str | None = None
    groups: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)


def detect_script(text: str) -> str:
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if dev and lat:
        return "mixed"
    if dev:
        return "devanagari"
    return "iast" if re.search(r"[āīūṛṝḷṃḥṅñṭḍṇśṣ]", text) else "english"


class QueryExpander:
    def __init__(self, csv_path: str | Path | None):
        self.groups: list[tuple[str, set[str], list[str], list[str]]] = []   # name, english, sanskrit, folded
        p = Path(csv_path) if csv_path else None
        if p and p.exists():
            with open(p, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    en = {w.strip().lower() for w in (r.get("english") or "").split("|") if w.strip()}
                    sa = [w.strip() for w in (r.get("sanskrit") or "").split("|") if w.strip()]
                    self.groups.append((r["group"], en, sa, [fold_for_search(w) for w in sa]))

    def analyse(self, query: str) -> QueryInfo:
        info = QueryInfo(query, detect_script(query))
        m = VERSE_RE.search(query)
        if m:
            info.verse = str(int(m.group(1).translate(str.maketrans("०१२३४५६७८९", "0123456789"))))
        words = fold_for_search(query).split()
        english = {w for w in re.findall(r"[a-z]+", query.lower())}
        english |= {w[:-1] for w in english if w.endswith("s") and len(w) > 3}
        for name, en, sa, folded in self.groups:
            hit = bool(en & english) or any(
                w.startswith(f) or (len(w) >= 4 and f.startswith(w)) for w in words for f in folded if len(f) >= 3)
            if hit:
                info.groups.append(name)
                if name not in GENERIC:          # "disease", "cause", "Vayu" match half the chapter: no help
                    info.added += [s for s in sa if s not in query]
        return info

    def expand(self, query: str) -> str:
        info = self.analyse(query)
        return query if not info.added else query + " " + " ".join(dict.fromkeys(info.added))


def content_words(text: str) -> set[str]:
    """Folded words that carry meaning (for overlap checks): length > 2, not English function words.
    Devanagari and Latin parts of a mixed sentence are folded separately, then cut to 6-letter stems."""
    stop = {"the", "and", "for", "are", "was", "this", "that", "with", "from", "what", "which", "does", "how",
            "why", "who", "its", "his", "her", "into", "verse", "shloka", "meaning", "explain", "mean", "here",
            "about", "there", "their", "they", "them", "has", "have", "had", "not", "but", "can", "will", "also",
            "word", "words", "say", "says", "said", "tell", "give", "please", "author", "acharya", "names",
            "name", "types", "type", "kinds", "kind", "many", "much", "when", "where", "these", "those"}
    out = set()
    for tok in re.findall(r"[ऀ-ॿ]+|[^\sऀ-ॿ]+", text):
        for w in fold_for_search(tok).split():
            if len(w) > 2 and w not in stop:
                out.add(w[:6])          # crude stem: the first 6 letters survive most Sanskrit endings
    return out


def folded_text(text: str) -> str:
    """The whole text folded (Devanagari and Latin parts separately), words kept whole."""
    return " " + " ".join(fold_for_search(t) for t in re.findall(r"[ऀ-ॿ]+|[^\sऀ-ॿ]+", text)) + " "


def soft_overlap(words: set[str], text: str) -> int:
    """How many stems in `words` occur in `text`. Stems of 4+ letters may sit inside a longer word, so
    sandhi-joined forms still count (अर्दित inside वक्त्रमर्दितं)."""
    f = folded_text(text)
    return sum(1 for w in words if (w in f if len(w) >= 4 else f" {w} " in f))