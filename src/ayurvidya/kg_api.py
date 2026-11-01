"""Scholar routes: upload knowledge-graph facts from the app (web/scholar.html). Switched on by
AYV_SCHOLAR_TOKEN. Facts live in the database (DATABASE_URL = Supabase, else data/study/notes.sqlite3);
after every upload or undo the graph is rebuilt in memory, so chat and the Vayu map change at once.

    GET  /scholar/template    CSV with the right columns and two example rows
    GET  /scholar/facts       all active facts (with who added them and the version)
    POST /scholar/upload      .csv or .xlsx -> {ok_rows, errors:[{row, problem}], version}
    POST /scholar/undo        remove the last upload
    GET  /scholar/export      the active facts as CSV (save it as data/kg/facts.csv for the paper)
All need the header X-Scholar: <AYV_SCHOLAR_TOKEN>.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import Response

from .kg_store import KGStore
from .notes import NotesDB

MAX_BYTES = 2 * 1024 * 1024
TEMPLATE = ("subject,relation,object,object_deva,verse,quote,status\n"
            "Prana,located_in,mouth,वक्त्र,13,यो वायुर्वक्त्रसञ्चारी स प्राणो नाम,approved\n"
            "Samana,causes,diarrhoea,अतीसार,17,गुल्माग्निसादातीसारप्रभृतीन्,approved\n")


def enable(app_state):
    """Connect the store, seed it from data/kg/facts.csv the first time, and use it for the live graph.
    Returns the router, or None when AYV_SCHOLAR_TOKEN is not set."""
    token = os.environ.get("AYV_SCHOLAR_TOKEN", "")
    if not token:
        return None
    s = app_state.settings
    db = NotesDB(os.environ.get("DATABASE_URL") or str(Path(s.data_dir) / "study" / "notes.sqlite3"))
    store = KGStore.for_app(db, app_state.store)
    store.seed_from_csv(Path(s.data_dir) / "kg" / "facts.csv")
    synonyms = Path(s.data_dir) / "synonyms.csv"

    def reload():
        g = store.graph(synonyms, app_state.kb) if s.use_kg else None
        app_state.graph = app_state.retriever.graph = app_state.chatbot.graph = g
        app_state.chatbot._cache.clear()

    reload()
    r = APIRouter(prefix="/scholar", tags=["scholar"])

    def check(x: str | None) -> str:
        if not x or not hmac.compare_digest(x, token):
            raise HTTPException(401, "Scholar access code required.")
        return "scholar"

    @r.get("/template")
    def template():
        return Response(TEMPLATE.encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="kg_facts_template.csv"'})

    @r.get("/facts")
    def facts(x_scholar: str | None = Header(default=None)):
        check(x_scholar)
        return {"version": store.version(), "facts": store.rows()}

    @r.post("/upload")
    async def upload(file: UploadFile = File(...), x_scholar: str | None = Header(default=None),
                     x_scholar_name: str | None = Header(default=None)):
        check(x_scholar)
        name = (file.filename or "").lower()
        if not name.endswith((".csv", ".xlsx")):
            raise HTTPException(400, "Upload a .csv or .xlsx file (use the template).")
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "The file is larger than 2 MB.")
        res = store.upload(data, name, (x_scholar_name or "scholar")[:60])
        if res["ok_rows"]:
            reload()
        return res

    @r.post("/undo")
    def undo(x_scholar: str | None = Header(default=None)):
        check(x_scholar)
        v = store.undo()
        reload()
        return {"removed_version": v}

    @r.get("/export")
    def export(x_scholar: str | None = Header(default=None)):
        check(x_scholar)
        return Response(store.export_csv().encode("utf-8-sig"), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="facts.csv"'})

    return r