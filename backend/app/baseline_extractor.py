"""
Baseline structured extractor — no external API calls.

Purpose: (1) lets the evaluation harness run entirely offline/reproducibly,
(2) acts as a graceful fallback if the Gemini API is unavailable or rate-limited,
(3) gives you a genuine "baseline vs. Gemini-enhanced" comparison for your resume
    metric (e.g. "LLM-based extraction improved match accuracy by X% over a
    keyword-matching baseline").

Uses a curated tech-skill vocabulary + regex matching for skills, and simple
heuristics for years of experience.
"""

import re

SKILL_VOCAB = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "react", "node.js", "nodejs", "express", "django", "flask", "fastapi",
    "spring boot", "sql", "postgresql", "mysql", "mongodb", "redis",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy",
    "docker", "kubernetes", "aws", "gcp", "azure", "ci/cd", "git", "github actions",
    "rest api", "graphql", "microservices", "system design", "data structures",
    "algorithms", "html", "css", "tailwind", "next.js", "vue", "angular",
    "spark", "hadoop", "airflow", "kafka", "elasticsearch", "linux", "bash",
    "llm", "generative ai", "prompt engineering", "rag", "langchain",
    "gemini api", "openai api", "huggingface", "transformers",
]

_EXPERIENCE_PATTERN = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience", re.IGNORECASE)


def extract_structured(text: str) -> dict:
    lower = text.lower()

    skills_found = sorted({skill for skill in SKILL_VOCAB if skill in lower})

    exp_match = _EXPERIENCE_PATTERN.search(lower)
    years_experience = int(exp_match.group(1)) if exp_match else 0

    if years_experience == 0:
        role_level = "intern" if "intern" in lower or "student" in lower else "unclear"
    elif years_experience <= 2:
        role_level = "junior"
    elif years_experience <= 5:
        role_level = "mid"
    else:
        role_level = "senior"

    achievement_pattern = re.compile(r"[^.\n]*\b\d{1,3}%[^.\n]*", re.IGNORECASE)
    key_achievements = [a.strip() for a in achievement_pattern.findall(text)][:5]

    return {
        "skills": skills_found,
        "years_experience": years_experience,
        "role_level": role_level,
        "key_achievements": key_achievements,
    }
