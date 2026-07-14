"""
Scoring engine.

Combines three signals into a final 0-100 match score:
  1. Skills overlap (weighted Jaccard-style) between resume and JD extracted skills
  2. Experience-level alignment (years + role level)
  3. Semantic similarity of the full texts (TF-IDF cosine) — catches relevant
     context that keyword matching alone misses (e.g. "led a team" implying
     leadership even if "leadership" isn't a listed skill)

Weights are tunable constants below — documented so you can defend the
design choice in an interview.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS_WEIGHT = 0.5
EXPERIENCE_WEIGHT = 0.2
SEMANTIC_WEIGHT = 0.3


def _skills_score(resume_skills, jd_skills) -> tuple[float, list, list]:
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    if not jd_set:
        return 100.0, list(resume_set), []

    matched = jd_set & resume_set
    missing = jd_set - resume_set

    score = (len(matched) / len(jd_set)) * 100
    return score, sorted(matched), sorted(missing)


def _experience_score(resume_years: int, jd_years: int) -> float:
    if jd_years == 0:
        return 100.0
    if resume_years >= jd_years:
        return 100.0
    # Linear falloff for under-experienced candidates
    return max(0.0, (resume_years / jd_years) * 100)


def _semantic_score(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim) * 100
    except ValueError:
        return 0.0


def compute_match(resume_text: str, jd_text: str, resume_struct: dict, jd_struct: dict) -> dict:
    skills_score, matched, missing = _skills_score(
        resume_struct.get("skills", []), jd_struct.get("skills", [])
    )
    experience_score = _experience_score(
        resume_struct.get("years_experience", 0), jd_struct.get("years_experience", 0)
    )
    semantic_score = _semantic_score(resume_text, jd_text)

    overall = (
        skills_score * SKILLS_WEIGHT
        + experience_score * EXPERIENCE_WEIGHT
        + semantic_score * SEMANTIC_WEIGHT
    )

    if overall >= 75:
        verdict = "Strong Fit"
    elif overall >= 50:
        verdict = "Moderate Fit"
    else:
        verdict = "Weak Fit"

    explanation = (
        f"Matched {len(matched)}/{len(matched) + len(missing)} required skills. "
        f"Experience alignment: {experience_score:.0f}%. "
        f"Contextual/semantic similarity: {semantic_score:.0f}%."
    )

    return {
        "overall_score": round(overall, 2),
        "skills_score": round(skills_score, 2),
        "experience_score": round(experience_score, 2),
        "semantic_score": round(semantic_score, 2),
        "verdict": verdict,
        "skill_gap": {"matched_skills": matched, "missing_skills": missing},
        "explanation": explanation,
    }
