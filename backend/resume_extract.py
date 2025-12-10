from io import BytesIO
import os
import json

import pdfplumber
from docx import Document
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF, DOCX, or plain text."""
    filename = filename.lower()

    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text

    if filename.endswith(".docx"):
        doc = Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    # Fallback: treat as plain text
    return file_bytes.decode("utf-8", errors="ignore")


def parse_resume_with_gemini(resume_text: str) -> dict:
    """Call Gemini and return a parsed JSON dict with resume fields."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    prompt = f"""
    You are a resume parsing engine.

    Extract the following fields from the resume text and return ONLY a valid JSON object:

    - full_name (string)
    - email (string)
    - phone (string)
    - total_experience_years (number, approximate)
    - current_role (string or null)
    - current_company (string or null)
    - location (string or null)
    - skills (array of strings)
    - education (array of objects: degree, institute, start_year, end_year)
    - experience (array of objects: job_title, company, start_date, end_date, description)

    If any field is missing, use null or empty array.

    Resume text:
    \"\"\"{resume_text}\"\"\"
    """

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    # ----- Call Gemini -----
    resp = requests.post(
        GEMINI_URL,
        headers=headers,
        data=json.dumps(body),
        timeout=60,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Print full response to the server console for debugging
        print("Gemini API error:", resp.status_code, resp.text)
        raise RuntimeError(
            f"Gemini API error {resp.status_code}: {resp.text}"
        ) from e

    data = resp.json()

    # Basic sanity checks
    if "candidates" not in data or not data["candidates"]:
        raise RuntimeError(f"No candidates in Gemini response: {data}")

    parts = data["candidates"][0]["content"]["parts"]
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    raw_text = "\n".join(text_parts).strip()

    # Gemini might wrap JSON in ```json ... ``` – strip that
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    # Extra safety: take substring between first '{' and last '}'
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"Gemini did not return JSON: {cleaned}")

    json_str = cleaned[start : end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from Gemini: {e}\nRaw: {json_str}"
        ) from e

    return parsed
