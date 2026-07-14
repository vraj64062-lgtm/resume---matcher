from pydantic import BaseModel, Field
from typing import List, Optional


class MatchRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text")
    job_description: str = Field(..., description="Raw job description text")


class SkillGap(BaseModel):
    missing_skills: List[str]
    matched_skills: List[str]


class MatchResponse(BaseModel):
    overall_score: float          # 0-100
    skills_score: float           # 0-100
    experience_score: float       # 0-100
    semantic_score: float         # 0-100
    verdict: str                  # "Strong Fit" | "Moderate Fit" | "Weak Fit"
    skill_gap: SkillGap
    explanation: str              # human-readable justification
    latency_ms: float
