# backend/main.py
import os
import traceback
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from db import get_connection
from resume_extract import extract_text_from_file, parse_resume_with_gemini

app = FastAPI(title="Resume Parser API")

# Read frontend URL from env; fall back to the Vercel URL you provided for convenience.
# In production prefer setting FRONTEND_URL in Railway Variables exactly to your Vercel domain.
DEFAULT_FRONTEND = "https://resume-parser-eosin.vercel.app"
FRONTEND_URL = os.environ.get("FRONTEND_URL", DEFAULT_FRONTEND)

if FRONTEND_URL and FRONTEND_URL != "*":
    allowed_origins = [FRONTEND_URL]
else:
    # during development you may set "*" but prefer exact domain in production
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def insert_parsed_resume(parsed: Dict[str, Any], filename: str) -> int:
    """
    Insert parsed resume into DB and return candidate_id.
    Expects parsed to be a dict with keys used below.
    """
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
            filename
        )
        cursor.execute(sql_candidate, candidate_values)
        candidate_id = cursor.lastrowid

        # Insert skills (if any)
        skills = parsed.get("skills") or []
        if skills and isinstance(skills, (list, tuple, set)):
            insert_skill_sql = "INSERT INTO skills (candidate_id, skill_name) VALUES (%s, %s)"
            for skill in skills:
                # guard: convert non-str to str
                cursor.execute(insert_skill_sql, (candidate_id, str(skill)))

        conn.commit()
        return candidate_id

    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accepts a file upload (PDF / DOCX / TXT), parses with Gemini, writes to DB,
    and returns parsed JSON and candidate_id.
    """
    try:
        # Read the uploaded file bytes
        file_bytes = await file.read()
        resume_text = extract_text_from_file(file_bytes, file.filename)

        # Parse using Gemini (this may raise descriptive runtime errors)
        parsed = parse_resume_with_gemini(resume_text)

        # Defensive: ensure parsed is a dict
        if not isinstance(parsed, dict):
            raise RuntimeError("Parsed resume returned non-dict result")

        # Insert into DB
        candidate_id = insert_parsed_resume(parsed, file.filename or "uploaded_file")

        return {"status": "success", "candidate_id": candidate_id, "parsed": parsed}

    except HTTPException:
        # Re-raise FastAPI HTTPExceptions unchanged
        raise
    except Exception as exc:
        # Print traceback to logs for debugging
        tb = traceback.format_exc()
        print("=== upload_resume ERROR ===")
        print(tb)
        # Return 500 with limited detail (useful during dev). In prod, you may want to return a generic message.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {str(exc)}"
        )


@app.get("/candidates")
def list_candidates():
    """
    Return list of candidates from DB. Uses dictionary cursor for nicer JSON.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as exc:
        tb = traceback.format_exc()
        print("=== list_candidates ERROR ===")
        print(tb)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"{type(exc).__name__}: {str(exc)}")


@app.get("/health")
def health():
    """
    Simple healthcheck — returns 200 if app can connect to DB.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return {"status": "ok"}
    except Exception as exc:
        print("=== health ERROR ===")
        print(traceback.format_exc())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="db-unreachable")
