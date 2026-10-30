"""HTTP routes for student notes (included by api.py when AYV_CLASS_CODE is set).

    GET  /notes/enabled          {"enabled": true}
    POST /notes/login            {class_code, name} -> {token, student_id, name}
    GET  /notes/template         the Excel template
    POST /notes/upload           multipart file (.xlsx, max 2 MB)  -> {ok_rows, errors, version}
    GET  /notes                  my active notes
    GET  /notes/verse/{id}       my notes on one verse
    POST /notes/undo             remove my last upload
    POST /notes/submit           send my notes to the teacher for review
All except login/template/enabled need the X-Student header (the token from login).
"""
from __future__ import annotations

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

MAX_BYTES = 2 * 1024 * 1024


class LoginIn(BaseModel):
    class_code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=80)


def make_router(app_state, chunks_by_id=None) -> APIRouter:
    notes = app_state.notes
    r = APIRouter(prefix="/notes", tags=["notes"])

    def who(token: str | None) -> str:
        sid = notes.verify(token) if token else None
        if not sid:
            raise HTTPException(401, "Please sign in with your class code first.")
        return sid

    @r.get("/enabled")
    def enabled():
        return {"enabled": True}

    @r.post("/login")
    def login(body: LoginIn):
        try:
            return notes.login(body.class_code, body.name)
        except PermissionError as e:
            raise HTTPException(403, str(e))

    @r.get("/template")
    def template():
        return Response(notes.template(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": 'attachment; filename="ayurvidya_notes_template.xlsx"'})

    @r.post("/upload")
    async def upload(file: UploadFile = File(...), x_student: str | None = Header(default=None)):
        sid = who(x_student)
        if not (file.filename or "").lower().endswith(".xlsx"):
            raise HTTPException(400, "Please upload an .xlsx Excel file (use the template).")
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "The file is larger than 2 MB.")
        return notes.upload(sid, data)

    @r.get("")
    def mine(x_student: str | None = Header(default=None)):
        return notes.list(who(x_student))

    @r.get("/verse/{shloka_id}")
    def verse(shloka_id: str, x_student: str | None = Header(default=None)):
        return notes.for_verse(who(x_student), shloka_id)

    @r.post("/undo")
    def undo(x_student: str | None = Header(default=None)):
        return {"removed_version": notes.undo(who(x_student))}

    @r.post("/submit")
    def submit(x_student: str | None = Header(default=None)):
        return {"submitted": notes.submit(who(x_student))}

    return r