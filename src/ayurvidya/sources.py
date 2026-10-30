"""Readable labels for the sources a chatbot answer cites (shown under the answer in the app)."""
from __future__ import annotations

from .schema import STEPS

STEP_NAMES = {n: name.capitalize() for n, name in STEPS.items()}


def verse_label(shloka_id: str) -> str:
    """'su.ni.1.13' -> 'Su. Ni. 1.13'"""
    parts = shloka_id.split(".")
    if len(parts) == 4:
        return f"{parts[0].capitalize()}. {parts[1].capitalize()}. {parts[2]}.{parts[3]}"
    return shloka_id


def source_label(cid: str, chunks_by_id: dict | None = None) -> str:
    """Turn a cited id into a short label a student understands."""
    if cid.startswith("step:"):                       # step:su.ni.1.3:5
        _, sid, n = cid.split(":")
        return f"{STEP_NAMES.get(int(n), 'Step ' + n)} of {verse_label(sid)}"
    if cid.startswith("note:"):                       # note:42
        return "Your notes"
    if cid.startswith("kg:"):
        return "Knowledge graph"
    if cid.startswith("mula."):                       # mula.su.ni.1.13
        return f"Sushruta Samhita, {verse_label(cid[5:])}"
    c = (chunks_by_id or {}).get(cid)
    if c is not None:
        where = ", ".join(verse_label(s) for s in c.shloka_ids[:2])
        return f"{c.work}" + (f" on {where}" if where else "")
    return cid