# backend/resume_extract.py

from io import BytesIO
import os
import json
import time
import re
from collections import OrderedDict

import pdfplumber
from docx import Document
import requests

# GEMINI KEY PROVIDED FROM ENV (Railway)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/"
    f"v1beta/models/{GEMINI_MODEL}:generateContent"
)


# ---------------------------
# FILE EXTRACTION
# ---------------------------
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


# ---------------------------
# SAFE JSON EXTRACTION HELPERS
# ---------------------------
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
    """Try strict JSON parse; raise detailed error on failure."""
    cleaned = maybe_text.strip()
    # remove ```json fences
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    json_str = _extract_json_from_text(cleaned)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"JSON decode error: {e}\nRaw JSON substring: {json_str}"
        ) from e


# ---------------------------
# GEMINI CALL WITH RETRIES
# ---------------------------
def _call_gemini_with_retries(prompt: str, attempts: int = 3, backoff: float = 1.0) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set!")

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
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
                GEMINI_URL, headers=headers, data=json.dumps(body), timeout=60
            )
            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            last_exception = e
            try:
                print(
                    f"Gemini call attempt {attempt} failed: {repr(e)}; response_text_snippet="
                    f"{resp.text[:500] if 'resp' in locals() and resp is not None else ''}"
                )
            except:
                pass

            if attempt < attempts:
                time.sleep(backoff * attempt)

    raise RuntimeError(f"Gemini call failed after {attempts} attempts: {last_exception}")


# ---------------------------
# TOLERANT PARSER FOR BROKEN JSON
# ---------------------------
def _extract_list_from_brackets(text, key_name):
    """
    Extract ["list","like","this"] inside key_name : [ ... ] even if malformed.
    """
    pattern = re.compile(
        rf'"{re.escape(key_name)}"\s*:\s*\[([^\]]*)\]',
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return []

    inner = m.group(1)

    # get quoted strings
    items = re.findall(r'"([^"]+)"', inner)

    # fallback: split on commas
    if not items:
        tokens = [t.strip() for t in inner.split(",") if t.strip()]
        items = [t.strip('" ').strip() for t in tokens if t.strip()]

    # remove duplicates while preserving order
    return list(OrderedDict.fromkeys(items))


def _extract_string_field(text, key_name):
    """Extract "key": "value" pattern."""
    pattern = re.compile(rf'"{re.escape(key_name)}"\s*:\s*"([^"]+)"', re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def tolerant_parse_raw_text(raw_text: str, resume_text: str) -> dict:
    """
    Best-effort extraction for badly formatted Gemini output.
    Returns dict shaped like your expected parsed result.
    """

    parsed = {
        "full_name": None,
        "email": None,
        "phone": None,
        "total_experience_years": None,
        "current_role": None,
        "current_company": None,
        "location": None,
        "skills": [],
        "education": [],
        "experience": [],
    }

    # full name
    parsed["full_name"] = _extract_string_field(raw_text, "full_name")
    if not parsed["full_name"]:
        # fallback: first non-empty line of resume
        for line in resume_text.splitlines():
            if line.strip():
                parsed["full_name"] = line.strip()
                break

    # email
    email_m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", raw_text)
    if not email_m:
        email_m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", resume_text)
    parsed["email"] = email_m.group(0) if email_m else None

    # phone
    phone_m = re.search(r"(\+?\d[\d\-\s]{6,}\d)", raw_text)
    if not phone_m:
        phone_m = re.search(r"(\+?\d[\d\-\s]{6,}\d)", resume_text)
    parsed["phone"] = phone_m.group(1) if phone_m else None

    # skills
    parsed["skills"] = _extract_list_from_brackets(raw_text, "skills")

    # experience years
    exp_m = re.search(r'"total_experience_years"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_text)
    if exp_m:
        parsed["total_experience_years"] = float(exp_m.group(1))

    # simple fields
    for field in ["current_role", "current_company", "location"]:
        parsed[field] = _extract_string_field(raw_text, field)

    # education extraction (basic)
    edu_pattern = re.compile(
        r'"degree"\s*:\s*"([^"]+)"\s*,\s*"institute"\s*:\s*"([^"]+)"',
        re.IGNORECASE,
    )
    edu_list = []
    for degree, institute in edu_pattern.findall(raw_text):
        edu_list.append({
            "degree": degree,
            "institute": institute,
            "start_year": None,
            "end_year": None,
        })
    parsed["education"] = edu_list

    return parsed


# ---------------------------
# MAIN GEMINI PARSER
# ---------------------------
def parse_resume_with_gemini(resume_text: str) -> dict:
    """Call Gemini → extract JSON → tolerant parse fallback."""

    prompt = f"""
You are a resume parsing engine.

Extract the following fields and return ONLY a valid JSON object:

- full_name
- email
- phone
- total_experience_years
- current_role
- current_company
- location
- skills (array)
- education (list of objects)
- experience (list of objects)

Resume text:
\"\"\"{resume_text}\"\"\"
"""

    # Call Gemini
    data = _call_gemini_with_retries(prompt)

    # extract text content
    text_content = None

    if "candidates" in data:
        try:
            content = data["candidates"][0]["content"]
            parts = content.get("parts") or []
            text_content = "\n".join(
                p.get("text", "") for p in parts if "text" in p
            ).strip()
        except:
            pass

    if not text_content and "outputs" in data:
        try:
            for out in data["outputs"]:
                content = out.get("content", {})
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    if parts:
                        text_content = "\n".join(
                            p.get("text", "") for p in parts if "text" in p
                        ).strip()
                    elif "text" in content:
                        text_content = content["text"].strip()
        except:
            pass

    if not text_content:
        raise RuntimeError("Gemini returned no textual response.")

    raw_text = text_content

    # TRY STRICT JSON
    try:
        parsed = _safe_parse_json_block(raw_text)
        return parsed
    except Exception as strict_err:
        print("Strict JSON parse failed:", strict_err)
        print("Using tolerant parser...")

    # TOLERANT PARSE
    try:
        fallback_parsed = tolerant_parse_raw_text(raw_text, resume_text)
        print("Tolerant parser succeeded.")
        return fallback_parsed
    except Exception as tolerant_err:
        print("Tolerant parser failed:", tolerant_err)

    # ULTIMATE FALLBACK
    print("Using ultimate fallback parser.")
    return fallback_parse_text(resume_text)


# ---------------------------
# VERY SIMPLE FALLBACK PARSER
# ---------------------------
_email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_phone_re = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")

def fallback_parse_text(resume_text: str) -> dict:
    """Minimal parser—guaranteed not to fail."""
    emails = _email_re.findall(resume_text) or []
    phones = _phone_re.findall(resume_text) or []

    first_line = next((l.strip() for l in resume_text.splitlines() if l.strip()), None)

    return {
        "full_name": first_line,
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "total_experience_years": None,
        "current_role": None,
        "current_company": None,
        "location": None,
        "skills": [],
        "education": [],
        "experience": [],
    }
