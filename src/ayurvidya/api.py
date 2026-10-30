"""HTTP API (FastAPI) for the web app.

    GET  /health                           cheap ping (use it from an uptime monitor to keep Render awake)
    GET  /chapters                         list chapters
    GET  /chapters/{chapter_id}/shlokas    list shlokas of a chapter
    GET  /shlokas/{shloka_id}              shloka + its 7 cached step outputs (instant, no LLM call)
    POST /chat                             {question, shloka_id?, history?} -> grounded answer + citations
    POST /chat/stream                      same, streamed as server-sent events (token ... done)
    GET  /graph                            the five Vayus from the knowledge graph (if data/kg/facts.csv exists)
    /notes/...                             student Excel notes (see notes_api.py, if present)
    POST /admin/shlokas/{shloka_id}/run    (re)generate the 7 steps for one shloka   [needs ADMIN_TOKEN]
    PUT  /admin/shlokas/{id}/steps/{n}     scholar review: approve or correct a step  [needs ADMIN_TOKEN]
    GET  /app/                             the web app (files in web/)

Run:  python scripts/serve.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import LLMError
from .schema import Shloka
from .service import build_app
from .sources import source_label

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
BUSY = "The AI is busy or today's free limit is used up. Please try again in a minute."

app_state = build_app()
chunks_by_id = {c.id: c for c in app_state.kb.chunks}
api = FastAPI(title="Ayurvidya RAG", version="2.0")


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    shloka_id: str | None = None
    history: list[tuple[str, str]] = []


class ReviewIn(BaseModel):
    status: str = Field(pattern="^(generated|reviewed|approved)$")
    note: str = ""
    output: str | None = None


def _admin(token: str | None) -> None:
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected or token != expected:
        raise HTTPException(401, "admin token required")


def _student(token: str | None) -> str | None:
    """X-Student header -> student id (only when student notes are enabled and the token is valid)."""
    if not token or app_state.notes is None:
        return None
    return app_state.notes.verify(token)


def _payload(a) -> dict:
    return {"answer": a.answer, "citations": a.citations, "grounded": a.grounded, "cached": a.cached,
            "sources": [{"id": c, "label": source_label(c, chunks_by_id)} for c in a.citations],
            "tags": {t: {"id": c, "label": source_label(c, chunks_by_id)} for t, c in a.tag_map.items()},
            "weak": [c["sentence"] for c in a.checks if c["status"] == "weak"],
            "timings": a.timings}


@api.get("/health")
def health():
    return {"ok": True, "chunks": len(app_state.kb.chunks), "chat_model": app_state.chatbot.llm.name}


@api.get("/chapters")
def chapters():
    return app_state.store.chapters()


@api.get("/chapters/{chapter_id}/shlokas")
def shlokas(chapter_id: str):
    return app_state.store.shlokas(chapter_id)


@api.get("/shlokas/{shloka_id}")
def shloka(shloka_id: str):
    sh = app_state.store.shloka(shloka_id)
    if not sh:
        raise HTTPException(404, "shloka not found")
    steps = app_state.store.steps(shloka_id)
    return {"shloka": sh, "steps": {n: {"output": d["output"], "status": d["status"], "valid": bool(d["valid"]),
                                        "refs": _step_refs(d["retrieved_ids"])} for n, d in steps.items()}}


def _step_refs(ids: list[str]) -> dict:
    """[R1], [R2] ... in a saved step = the non-example chunks it was written from, in order."""
    refs = [c for c in ids if c in chunks_by_id and chunks_by_id[c].source_type != "gold"]
    return {f"R{i}": {"id": c, "label": source_label(c, chunks_by_id)} for i, c in enumerate(refs, 1)}


@api.get("/source/{cid:path}")
def source(cid: str):
    """The text behind a citation, for the footnote sheet in the app."""
    if cid.startswith("step:"):
        _, sid, n = cid.split(":")
        d = app_state.store.steps(sid).get(int(n))
        if not d:
            raise HTTPException(404, "not found")
        return {"id": cid, "label": source_label(cid), "text": d["output"]}
    c = chunks_by_id.get(cid)
    if c is None:
        raise HTTPException(404, "not found")
    return {"id": cid, "label": source_label(cid, chunks_by_id), "text": c.text, "type": c.source_type}


@api.post("/chat")
def chat(body: ChatIn, x_student: str | None = Header(default=None)):
    try:
        a = app_state.chatbot.ask(body.question, body.shloka_id, body.history, _student(x_student))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except LLMError:
        raise HTTPException(503, BUSY)
    return _payload(a)


@api.post("/chat/stream")
def chat_stream(body: ChatIn, x_student: str | None = Header(default=None)):
    if body.shloka_id and not app_state.store.shloka(body.shloka_id):
        raise HTTPException(404, "shloka not found")
    student = _student(x_student)

    def events():
        try:
            for kind, value in app_state.chatbot.stream(body.question, body.shloka_id, body.history, student):
                if kind == "token":
                    yield f"data: {json.dumps({'type': 'token', 'text': value}, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'done', **_payload(value)}, ensure_ascii=False)}\n\n"
        except LLMError:
            yield f"data: {json.dumps({'type': 'error', 'message': BUSY})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api.get("/graph")
def graph():
    if app_state.graph is None:
        raise HTTPException(404, "no knowledge graph yet (data/kg/facts.csv)")
    return app_state.graph.summary()


@api.post("/admin/shlokas/{shloka_id}/run")
def run(shloka_id: str, force: bool = False, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    row = app_state.store.shloka(shloka_id)
    if not row:
        raise HTTPException(404, "shloka not found")
    sh = Shloka(id=row["id"], samhita="sushruta", sthana="nidana", chapter=int(row["chapter_id"].split(".")[-1]),
                number=row["number"], text=row["text"], is_prose=bool(row["is_prose"]), variant=row["variant"])
    res = app_state.orchestrator.run_shloka(sh, force=force)
    return {n: {"valid": r.valid, "issues": r.issues, "attempts": r.attempts} for n, r in res.items()}


@api.put("/admin/shlokas/{shloka_id}/steps/{step}")
def review(shloka_id: str, step: int, body: ReviewIn, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    app_state.store.set_review(shloka_id, step, body.status, body.note, body.output)
    return {"ok": True}


try:                                   # student notes (Excel upload) - added in its own commit
    from .notes_api import make_router
    if app_state.notes is not None:
        api.include_router(make_router(app_state, chunks_by_id))
except ImportError:
    pass


# ---------------------------------------------------------------- web app
@api.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/app/")


if WEB_DIR.exists():
    api.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="app")