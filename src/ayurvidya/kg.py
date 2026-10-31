"""Knowledge graph for Su. Ni. 1: the five Vayus -> seat, actions, disorders, avarana (covered by Pitta /
Kapha), each fact tied to the verse that states it (with an exact quote).

data/kg/facts.csv  columns: subject, relation, object, object_deva, verse, quote, status
    status: seed (written by hand from the verse, still to be checked) | draft (LLM-extracted) |
            approved (checked by a scholar). Only seed + approved are loaded; drafts wait for review.

How it answers a question (graph RAG):
  1. link: find the Vayus the question names (any script, via data/synonyms.csv) and the objects whose
     words it uses ("hiccup", "श्वास", "colon") and the kind of relation it asks for (where / does / causes);
  2. walk: facts of the linked Vayus (1 hop) + facts whose object matches, then the other facts of THOSE
     Vayus (2 hops: "which Vayu causes hiccup" -> Prana -> where Prana lives);
  3. use: the best facts go into the chat prompt as [G#] lines, and their verses become a third ranked
     list in the RRF fusion, next to BM25 and dense search.
Plain Python dicts: the graph is a few hundred facts, no graph database needed.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .query import QueryExpander, content_words, folded_text, soft_overlap
from .text import fold_for_search

VAYUS = ["Prana", "Udana", "Samana", "Vyana", "Apana"]
DEVA = {"Prana": "प्राणः", "Udana": "उदानः", "Samana": "समानः", "Vyana": "व्यानः", "Apana": "अपानः", "Vayu": "वायुः"}
REL_WORDS = {
    "located_in": {"where", "seat", "seats", "located", "location", "live", "lives", "moves", "move", "site", "place"},
    "performs": {"do", "does", "function", "functions", "action", "actions", "work", "role", "normally", "help", "helps"},
    "causes": {"disease", "diseases", "disorder", "disorders", "cause", "causes", "angered", "vitiated", "aggravated"},
    "covered_by_pitta": {"pitta", "covered", "joined", "avarana", "avrita"},
    "covered_by_kapha": {"kapha", "covered", "joined", "avarana", "avrita"},
}
REL_LABEL = {"located_in": "seat", "performs": "does", "causes": "causes when vitiated",
             "covered_by_pitta": "when covered by Pitta", "covered_by_kapha": "when covered by Kapha"}


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    object_deva: str
    verse: str           # shloka id, e.g. su.ni.1.13
    quote: str
    status: str

    def line(self) -> str:
        return (f"{self.subject} ({DEVA.get(self.subject, '')}) {REL_LABEL.get(self.relation, self.relation)}: "
                f"{self.object} ({self.object_deva}) — Su. Ni. 1.{self.verse.split('.')[-1]}, «{self.quote}»")


class KnowledgeGraph:
    def __init__(self, facts: list[Fact], expander: QueryExpander | None = None, kb=None):
        self.facts = facts
        self.expander = expander
        self.by_subject: dict[str, list[int]] = {}
        for i, f in enumerate(facts):
            self.by_subject.setdefault(f.subject, []).append(i)
        # words that name a KIND of thing ("disease", "रोग", "seat") say nothing about WHICH fact is meant
        generic = set().union(*REL_WORDS.values())
        if expander is not None:
            for name, en, sa, _ in expander.groups:
                if name in ("disease", "cause", "sign", "seat", "action", "vata", "five", "vitiated"):
                    generic |= content_words(" ".join(en) + " " + " ".join(sa))
        generic = {w[:6] for w in generic}
        self.obj_words = [content_words(f"{f.object} {f.object_deva}") - generic for f in facts]
        self.verse_chunk = {}
        if kb is not None:
            for f in facts:
                i = kb.by_id.get(f"mula.{f.verse}")
                if i is not None:
                    self.verse_chunk[f.verse] = i
        # aliases of each subject: its synonym group (English + Sanskrit forms), folded
        self.alias: dict[str, set[str]] = {s: {s.lower()} for s in self.by_subject}
        if expander is not None:
            for name, en, _, folded in expander.groups:
                for s in self.by_subject:
                    if name == s.lower() or (s == "Vayu" and name == "vata"):
                        self.alias[s] |= en | {f for f in folded if len(f) >= 3}

    @classmethod
    def load(cls, path: str | Path, synonyms: str | Path | None = None, kb=None, statuses=("seed", "approved")):
        facts = []
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("status", "seed") not in statuses:
                    continue
                v = r["verse"].strip()
                facts.append(Fact(r["subject"].strip(), r["relation"].strip(), r["object"].strip(),
                                  (r.get("object_deva") or "").strip(), v if v.startswith("su.") else f"su.ni.1.{v}",
                                  (r.get("quote") or "").strip(), r.get("status", "seed")))
        return cls(facts, QueryExpander(synonyms) if synonyms else None, kb)

    # ---------------------------------------------------------- linking + walking
    def link(self, question: str) -> tuple[set[str], set[str]]:
        words = set(re.findall(r"[a-z]+", question.lower()))
        folded = folded_text(question)
        subjects = {s for s, al in self.alias.items()
                    if (al & words) or any(len(a) >= 4 and f" {a}" in folded for a in al if not a.isascii() or a not in words)}
        subjects |= {s for s in self.by_subject if fold_for_search(DEVA.get(s, s))[:4] in folded}
        rels = {r for r, ws in REL_WORDS.items() if ws & words}
        return subjects, rels

    def score_facts(self, question: str, shloka_id: str | None = None) -> list[tuple[float, int]]:
        q = self.expander.expand(question) if self.expander else question
        subjects, rels = self.link(q)
        qtext = folded_text(q)
        scores: dict[int, float] = {}
        for i, f in enumerate(self.facts):
            s = 0.0
            if f.subject in subjects:
                s += 1 if f.subject == "Vayu" else 3          # "Vayu" alone is too general to decide
            ov = soft_overlap(self.obj_words[i], qtext) if self.obj_words[i] else 0
            s += 2 * min(ov, 2)
            if f.relation in rels:
                s += 1.5 if (f.subject in subjects or ov) else 0.5
            if shloka_id and f.verse == shloka_id:
                s += 1
            if s >= 2:
                scores[i] = s
        # a question about all of them ("the five Vayus", "each Vayu") gets that relation for every Vayu
        if {"five", "all", "each", "every", "names", "types", "kinds"} & set(re.findall(r"[a-z]+", q.lower())):
            for i, f in enumerate(self.facts):
                if f.subject in VAYUS and f.relation in (rels or {"located_in"}):
                    scores[i] = max(scores.get(i, 0), 2.5)
        # 2nd hop: Vayus reached through a matching object bring their other facts (lower weight)
        hop = {self.facts[i].subject for i, s in scores.items() if s >= 2 and self.facts[i].subject not in subjects
               and soft_overlap(self.obj_words[i], qtext)}
        for sub in hop:
            for i in self.by_subject.get(sub, []):
                if i not in scores:
                    f = self.facts[i]
                    scores[i] = 1.0 + (1.0 if f.relation in rels else 0)
        return sorted(((s, i) for i, s in scores.items()), key=lambda x: (-x[0], x[1]))

    def facts_for(self, question: str, shloka_id: str | None = None, k: int = 6) -> list[tuple[str, str]]:
        return [(self.facts[i].line(), self.facts[i].verse) for _, i in self.score_facts(question, shloka_id)[:k]]

    def rank_chunks(self, query: str, kb, allowed=None) -> list[int]:
        """Verses reached through the graph, as KB chunk indices for RRF. A verse's rank = the summed score
        of its matching facts, so a verse stating three asked-for facts beats one stating a single fact."""
        scored = self.score_facts(query)
        if not scored or scored[0][0] < 3.5:   # the graph only votes when it clearly recognises the question
            return []
        per_verse: dict[str, float] = {}
        for sc, i in scored:
            per_verse[self.facts[i].verse] = per_verse.get(self.facts[i].verse, 0) + sc
        out = []
        for v, _ in sorted(per_verse.items(), key=lambda x: -x[1]):
            c = self.verse_chunk.get(v, kb.by_id.get(f"mula.{v}"))
            if c is not None and (allowed is None or allowed[c]):
                out.append(c)
        return out[:20]

    # ---------------------------------------------------------- for the app's Vayu map
    def summary(self) -> dict:
        def items(sub, rel):
            return [{"text": f"{f.object} ({f.object_deva})", "verse": f.verse}
                    for f in (self.facts[i] for i in self.by_subject.get(sub, [])) if f.relation == rel]
        vayus = []
        for v in VAYUS:
            if v not in self.by_subject:
                continue
            other = [{"text": f"{REL_LABEL.get(f.relation, f.relation)}: {f.object}", "verse": f.verse}
                     for f in (self.facts[i] for i in self.by_subject[v]) if f.relation.startswith("covered")]
            vayus.append({"name": v, "deva": DEVA[v], "sthana": items(v, "located_in"), "karma": items(v, "performs"),
                          "vyadhi": items(v, "causes"), "other": other})
        return {"n_facts": len(self.facts), "checked": sum(f.status == "approved" for f in self.facts), "vayus": vayus}