import os
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import MatchRequest, MatchResponse
from app import scorer, baseline_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-matcher")

app = FastAPI(
    title="AI Resume-to-Job Matcher",
    description="Explainable resume/JD matching using LLM-based extraction + weighted scoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))
if USE_GEMINI:
    from app import gemini_client


def _extract(text: str) -> dict:
    if USE_GEMINI:
        try:
            return gemini_client.extract_structured(text)
        except Exception as e:
            logger.warning(f"Gemini extraction failed, falling back to baseline: {e}")
    return baseline_extractor.extract_structured(text)


@app.get("/health")
def health():
    return {"status": "ok", "using_gemini": USE_GEMINI}


@app.post("/match", response_model=MatchResponse)
def match(request: MatchRequest):
    if not request.resume_text.strip() or not request.job_description.strip():
        raise HTTPException(status_code=400, detail="resume_text and job_description cannot be empty.")

    start = time.perf_counter()

    resume_struct = _extract(request.resume_text)
    jd_struct = _extract(request.job_description)

    result = scorer.compute_match(
        request.resume_text, request.job_description, resume_struct, jd_struct
    )

    latency_ms = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(latency_ms, 2)

    return result
