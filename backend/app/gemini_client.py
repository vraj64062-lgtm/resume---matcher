"""
Gemini API wrapper.

Uses Gemini to extract structured data (skills, years of experience, role level,
key achievements) from raw resume and job-description text. Structured JSON
output means downstream scoring is deterministic and testable instead of
relying on the LLM to also "judge" the match (which is harder to evaluate).

Requires GEMINI_API_KEY to be set as an environment variable.
Get a key at: https://aistudio.google.com/apikey
"""

import os
import json
import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

EXTRACTION_PROMPT = """You are a structured-data extraction engine. Given the text below,
extract ONLY the following as strict JSON, with no markdown fences, no preamble:

{{
  "skills": ["list", "of", "technical", "skills", "mentioned"],
  "years_experience": <number, estimate if not explicit, 0 if entry-level/none stated>,
  "role_level": "<intern|junior|mid|senior|unclear>",
  "key_achievements": ["short phrases of quantifiable achievements, empty list if none"]
}}

TEXT:
{text}
"""


def _get_model():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a key from https://aistudio.google.com/apikey "
            "and set it as an environment variable before running."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")


def extract_structured(text: str) -> dict:
    """Calls Gemini to turn raw resume/JD text into structured JSON."""
    model = _get_model()
    prompt = EXTRACTION_PROMPT.format(text=text[:8000])  # guard against huge inputs
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Gemini sometimes wraps JSON in ```json fences despite instructions — strip defensively
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1) if raw.startswith("json\n") else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: return an empty structure rather than crashing the request
        return {"skills": [], "years_experience": 0, "role_level": "unclear", "key_achievements": []}
