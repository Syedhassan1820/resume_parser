# backend/resume_extract.py
from io import BytesIO
import os
import json
import time
import re

import pdfplumber
from docx import Document
import requests

# Do NOT call load_dotenv() in production; Railway / Railway variables populate os.environ.
# If you test locally and want to load a .env file, uncomment the next two lines and keep .env in .gitignore:
# from dotenv import load_dotenv
# load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from PDF, DOCX, or plain text."""
    filename = (filename or "").lower()

    if filename.endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        except Exception as e:
            # If pdfplumber fails, fall back to plain text decode
            print("pdfplumber error:", e)
            try:
                return file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return text

    if filename.endswith(".docx"):
        try:
            doc = Document(BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print("python-docx error:", e)
            try:
                return file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                return ""

    # Fallback: treat as plain text
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_json_from_text(maybe_text: str) -> str:
    """Return the first {...} substring from maybe_text (or raise)."""
    if not maybe_text:
        raise RuntimeError("No text available to extract JSON from.")

    start = maybe_text.find("{")
    end = maybe_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Could not find JSON object boundaries in text.")
    return maybe_text[start : end + 1]


def _safe_parse_json_block(maybe_text: str) -> dict:
    """
    Try to find and parse first JSON object inside maybe_text.
    Returns parsed dict or raises RuntimeError with helpful content.
    """
    cleaned = maybe_text.strip()
    # Remove surrounding markdown code fences like ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    # Now extract {...}
    json_str = _extract_json_from_text(cleaned)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Provide the raw JSON substring when failing for easier debugging
        raise RuntimeError(f"JSON decode error: {e}\nRaw JSON substring: {json_str}") from e


def _call_gemini_with_retries(prompt: str, attempts: int = 3, backoff: float = 1.0) -> dict:
    """Call the Gemini API with retries and return parsed JSON response (not final parsed object)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set. Set GEMINI_API_KEY in Railway / env.")

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

    last_exception = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                GEMINI_URL,
                headers=headers,
                data=json.dumps(body),
                timeout=60,
            )
            # If status is not 2xx, raise for status to handle in except below
            resp.raise_for_status()
            # Return full parsed JSON body (may contain nested text parts we handle later)
            try:
                return resp.json()
            except Exception as e:
                # If resp.json() fails, raise useful information
                text = resp.text
                raise RuntimeError(f"Failed to decode JSON response from Gemini: {e}\nRaw text: {text}") from e

        except Exception as e:
            last_exception = e
            # Print helpful debug info for logs; include response text if available
            try:
                resp_text = resp.text if 'resp' in locals() and resp is not None else "<no response text>"
            except Exception:
                resp_text = "<error reading response text>"
            print(f"Gemini call attempt {attempt} failed: {repr(e)}; response_text_snippet={resp_text[:1000]}")
            if attempt < attempts:
                time.sleep(backoff * attempt)
            else:
                # All attempts failed
                raise RuntimeError(f"Gemini call failed after {attempts} attempts: {last_exception}") from last_exception


def parse_resume_with_gemini(resume_text: str) -> dict:
    """
    Call Gemini and return a parsed JSON dict with resume fields.
    This function is defensive: it prints diagnostics and raises informative errors.
    """
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

    # Call Gemini
    data = _call_gemini_with_retries(prompt)

    # The exact response shape may vary. Attempt to get text parts safely.
    # Older code expected data["candidates"][0]["content"]["parts"]
    text_candidates = None
    try:
        if isinstance(data, dict) and "candidates" in data:
            # keep backward compatibility
            cand = data.get("candidates") or []
            if cand:
                content = cand[0].get("content") or {}
                parts = content.get("parts") or []
                text_candidates = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
    except Exception as e:
        print("Error extracting candidates parts:", e)

    if not text_candidates:
        # Try other common keys (best-effort)
        try:
            if isinstance(data, dict) and "outputs" in data:
                outputs = data.get("outputs") or []
                # outputs might contain ['content']['text'] or similar
                for out in outputs:
                    if isinstance(out, dict):
                        content = out.get("content") or {}
                        if isinstance(content, dict):
                            # try nested parts
                            parts = content.get("parts") or []
                            if parts:
                                text_candidates = "\n".join(p.get("text", "") for p in parts if "text" in p).strip()
                            elif "text" in content:
                                text_candidates = content.get("text", "").strip()
        except Exception as e:
            print("Error extracting outputs parts:", e)

    # If still empty, attempt to stringify top-level keys for debug
    if not text_candidates:
        print("Gemini raw response (full):", json.dumps(data)[:2000])
        raise RuntimeError(f"No textual candidate parts found in Gemini response. See logs for raw response.")

    # Now we have a textual block (likely containing JSON). Try to extract JSON object.
    raw_text = text_candidates
    try:
        parsed = _safe_parse_json_block(raw_text)
        # final sanity: ensure keys exist (but do not enforce strict typing)
        if not isinstance(parsed, dict):
            raise RuntimeError("Parsed Gemini output is not a JSON object/dict.")
        return parsed
    except Exception as e:
        # Log the raw_text for debugging and show helpful error
        print("Failed to parse JSON from Gemini raw_text snippet:", raw_text[:2000])
        raise RuntimeError(f"Failed to parse JSON from Gemini response: {e}") from e


# --- Small fallback parser if Gemini is unavailable (VERY simple) ---
_email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_phone_re = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")

def fallback_parse_text(resume_text: str) -> dict:
    """
    Extremely simple fallback parser to extract a couple of fields.
    Use this only for testing / graceful degradation.
    """
    emails = _email_re.findall(resume_text) or []
    phones = _phone_re.findall(resume_text) or []
    # rough "name" heuristic: first non-empty line
    first_line = ""
    for line in resume_text.splitlines():
        l = line.strip()
        if l:
            first_line = l
            break

    return {
        "full_name": first_line or None,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "total_experience_years": None,
        "current_role": None,
        "current_company": None,
        "location": None,
        "skills": [],
        "education": [],
        "experience": []
    }
