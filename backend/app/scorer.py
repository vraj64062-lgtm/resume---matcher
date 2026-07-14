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


def _skills_score(resume_skills, jd_skills, semantic_score: float) -> tuple[float, list, list, bool]:
    """
    Returns (score, matched, missing, low_confidence).

    IMPORTANT: if the JD has no skills our vocabulary can detect (e.g. a very
    short or non-technical JD like "mechanical engineer"), we do NOT default
    to a perfect 100% — that previously produced false "Moderate Fit" results
    for obviously mismatched pairs (e.g. an AI/ML resume vs a one-line
    unrelated JD), since "no detected requirements" was being read as
    "no requirements to fail." Instead we fall back to the semantic similarity
    signal, which correctly reflects how unrelated the texts actually are,
    and flag the result as low_confidence so the caller can be transparent
    about it.
    """
    resume_set = set(s.lower() for s in resume_skills)
    jd_set = set(s.lower() for s in jd_skills)

    if not jd_set:
        return semantic_score, list(resume_set), [], True

    matched = jd_set & resume_set
    missing = jd_set - resume_set

    score = (len(matched) / len(jd_set)) * 100
    return score, sorted(matched), sorted(missing), False


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
    # Semantic score is computed first since it's used as the fallback signal
    # when the JD has no detectable required skills (see _skills_score).
    semantic_score = _semantic_score(resume_text, jd_text)

    skills_score, matched, missing, low_confidence = _skills_score(
        resume_struct.get("skills", []), jd_struct.get("skills", []), semantic_score
    )
    experience_score = _experience_score(
        resume_struct.get("years_experience", 0), jd_struct.get("years_experience", 0)
    )

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

    if low_confidence:
        explanation = (
            f"The job description didn't contain specific recognizable skill "
            f"keywords, so this score is based mainly on overall text similarity "
            f"({semantic_score:.0f}%) rather than a skill-by-skill match. "
            f"Treat this result with lower confidence and consider providing a "
            f"more detailed job description for a more reliable score."
        )
    else:
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
        "low_confidence": low_confidence,
    }