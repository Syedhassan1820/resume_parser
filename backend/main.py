from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from db import get_connection
from resume_extract import extract_text_from_file, parse_resume_with_gemini

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def insert_parsed_resume(parsed: dict, filename: str) -> int:
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

        # Insert skills
        for skill in parsed.get("skills") or []:
            cursor.execute(
                "INSERT INTO skills (candidate_id, skill_name) VALUES (%s, %s)",
                (candidate_id, skill),
            )

        conn.commit()
        return candidate_id

    finally:
        cursor.close()
        conn.close()


@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        resume_text = extract_text_from_file(file_bytes, file.filename)
        parsed = parse_resume_with_gemini(resume_text)
        candidate_id = insert_parsed_resume(parsed, file.filename)
        return {"status": "success", "candidate_id": candidate_id, "parsed": parsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/candidates")
def list_candidates():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
