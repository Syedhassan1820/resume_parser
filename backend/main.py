# backend/main.py
import os
import traceback
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from db import get_connection
from resume_extract import (
    extract_text_from_file,
    parse_resume_with_gemini,
    tolerant_parse_raw_text,
    fallback_parse_text,
)

app = FastAPI(title="Resume Parser API")


# ----------------------------
# CORS CONFIGURATION
# ----------------------------
allowed_origins = [
    "https://resume-parser-eosin.vercel.app",   # Your Vercel frontend
    "https://resumeparser-backend.up.railway.app",  # Backend itself
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# DB INSERTION
# ----------------------------
def insert_parsed_resume(parsed: Dict[str, Any], filename: str) -> int:
    """Insert parsed resume into DB and return candidate_id."""

    if not isinstance(parsed, dict):
        raise ValueError("Parsed resume is not a dict")

    conn = get_connection()
    cursor = conn.cursor()

    try:
        sql_candidate = """
            INSERT INTO candidates
            (full_name, email, phone, total_experience_years,
             current_role, current_company, location, resume_file_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        candidate_values = (
            parsed.get("full_name"),
            parsed.get("email"),
            parsed.get("phone"),
            parsed.get("total_experience_years"),
            parsed.get("current_role"),
            parsed.get("current_company"),
            parsed.get("location"),
            filename,
        )

        cursor.execute(sql_candidate, candidate_values)
        candidate_id = cursor.lastrowid

        # Insert skills (safe conversion)
        skills = parsed.get("skills") or []
        if isinstance(skills, (list, tuple, set)):
            for skill in skills:
                cursor.execute(
                    "INSERT INTO skills (candidate_id, skill_name) VALUES (%s, %s)",
                    (candidate_id, str(skill)),
                )

        conn.commit()
        return candidate_id

    finally:
        cursor.close()
        conn.close()


# ----------------------------
# UPLOAD ENDPOINT
# ----------------------------
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Handles file upload → extraction → Gemini parsing → fallback parsing → DB insert.
    NEVER returns 500 to frontend unless something extremely unexpected happens.
    """

    try:
        file_bytes = await file.read()
        resume_text = extract_text_from_file(file_bytes, file.filename)

        # -------------------------
        # Try Gemini → tolerant → fallback
        # -------------------------
        parsed = None

        try:
            parsed = parse_resume_with_gemini(resume_text)
        except Exception as gemini_error:
            print("\n=== GEMINI ERROR (fallback triggered) ===")
            print(gemini_error)
            print("========================================")

            try:
                parsed = tolerant_parse_raw_text(str(gemini_error), resume_text)
                print("Tolerant parser succeeded.")
            except Exception as tol_err:
                print("Tolerant parser failed:", tol_err)
                print("Using ultimate fallback parser.")
                parsed = fallback_parse_text(resume_text)

        # Safety: ensure parsed object is dict
        if not isinstance(parsed, dict):
            parsed = fallback_parse_text(resume_text)

        # -------------------------
        # Insert into DB
        # -------------------------
        candidate_id = insert_parsed_resume(parsed, file.filename or "uploaded_file")

        return {
            "status": "success",
            "candidate_id": candidate_id,
            "parsed": parsed,
        }

    except Exception as exc:
        print("=== UNEXPECTED upload_resume ERROR ===")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(exc),
        }


# ----------------------------
# LIST CANDIDATES
# ----------------------------
@app.get("/candidates")
def list_candidates():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as exc:
        print("=== list_candidates ERROR ===")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {str(exc)}",
        )


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return {"status": "ok"}
    except Exception:
        return {"status": "db-error"}
