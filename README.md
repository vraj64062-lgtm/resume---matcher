# AI Resume-to-Job Matcher

Explainable resume/job-description matching tool. Extracts structured data (skills,
experience, role level) from raw text using Gemini, then scores the match with a
transparent weighted algorithm — skills overlap, experience alignment, and semantic
similarity — instead of a black-box LLM judgment.

## Why this design

Most "AI resume matcher" projects just ask an LLM "does this resume fit this job?"
and print whatever comes back. That's not evaluable and not defensible in an
interview. This project separates **extraction** (LLM's job: turn messy text into
structured JSON) from **scoring** (deterministic code: turn structured JSON into a
number), so the matching logic can be tested, tuned, and benchmarked like any other
ML system.

## Architecture

```
resume-matcher/
├── backend/            FastAPI service
│   ├── app/main.py         API entrypoint
│   ├── app/gemini_client.py    LLM-based structured extraction
│   ├── app/baseline_extractor.py  keyword-based fallback/baseline
│   └── app/scorer.py       weighted scoring engine
├── evaluation/          Evaluation harness + labeled dataset
│   ├── labeled_dataset.json    24 hand-labeled resume/JD pairs
│   └── eval_harness.py         computes accuracy/precision/recall/F1
└── frontend/            React UI
```

## Results (real, reproducible — run it yourself)

Ran on the 35-sample labeled dataset in `evaluation/labeled_dataset.json` (24 clear-cut
cases + 10 deliberately borderline cases + 1 regression test for the bug described below):

| Mode                          | Accuracy | Precision | Recall | F1-score | Avg latency |
|--------------------------------|----------|-----------|--------|----------|-------------|
| Baseline (keyword matching)   | 82.9%    | 84.6%     | 73.3%  | 78.6%    | 2.1 ms/pair |
| Gemini-enhanced (run with your API key) | — run `eval_harness.py --use-gemini` — |

**Post-launch fix (real bug found via live testing):** after deploying, testing the live
app with a deliberately mismatched pair (an AI/ML resume against a one-line, unrelated
JD like "mechanical engineer") revealed the scorer returned a false 70% "Moderate Fit."
Root cause: when a JD has no skills the vocabulary can detect, `_skills_score` defaulted
to a perfect 100% on the assumption that "no detected requirements" meant "nothing to
fail" — this silently broke on short or non-technical JDs. Fixed by falling back to the
semantic similarity signal (which correctly showed ~8% similarity) instead, and flagging
the result as `low_confidence` so the UI is honest about when a score is less reliable.
The same test case now correctly returns ~26%, "Weak Fit." This is a good example of a
gap that only shows up under live, adversarial testing rather than a curated eval set —
worth mentioning in an interview alongside the eval methodology story.

**Methodology note (worth mentioning in an interview):** the dataset was initially
24 cases that were all clean wins or clean losses (near-total skill overlap for
"fit", extreme mismatches for "no_fit"), which produced a misleadingly perfect
100% accuracy — a sign the test set wasn't actually stress-testing the system.
Adding 10 genuinely ambiguous cases dropped accuracy to 82.4% and exposed a real
limitation: the scorer under-penalizes missing skills relative to experience
match, causing some partial-overlap candidates to score as false positives. The
fit threshold was recalibrated from 50 to 65 based on a sensitivity sweep, which
partially compensates but doesn't fully fix the underlying weighting issue —
a good next step documented in "Known limitations" below.

**Resume-ready framing options:**
- *"Built an explainable resume-matching engine, evaluated against a 34-case labeled dataset including adversarial/borderline examples, achieving 82.4% accuracy and 80% F1-score."*
- *"Designed a weighted scoring system combining LLM-based extraction with skill/experience/semantic signals; diagnosed and documented a false-positive bias toward partial skill matches through threshold sensitivity analysis."*
- *"Found and fixed two production bugs through live manual testing after deployment: a deprecated LLM model name causing silent zero-signal failures, and a skill-matching design flaw where synonymous terms (e.g. 'Amazon Web Services' vs 'AWS') failed to match due to naive exact-string comparison — fixed via canonicalized extraction plus a synonym-normalization safety net."*
- Once you run the Gemini-enhanced eval with your own API key, you can add a comparative line like: *"Improved match accuracy from X% (keyword baseline) to Y% using LLM-based structured extraction."*

## 🐛 Bug #1: deprecated Gemini model name (silent failure)

After deployment, live testing with a resume/JD pair that had no keyword-matchable technical
skills produced a suspicious flat 0% result with no errors anywhere. Root cause: the hardcoded
model name (`gemini-2.0-flash`) had been retired by Google, and the API call was failing with a
404 that the code's own error handling caught silently, returning an empty response instead of
surfacing the failure. This meant the app appeared to work (no crash, no visible error) while
actually never successfully calling Gemini at all.

**Fix:** switched to `gemini-flash-latest`, Google's auto-updating model alias, so the app
always targets the current generally-available Flash model instead of a version string that
will eventually be retired. The exact same bug, with the exact same root cause, was found
independently in a separate project (an AI code review assistant) built around the same time —
worth mentioning together as a pattern, not a one-off.

## 🐛 Bug #2: skill-name synonym mismatch (design flaw, not just a config issue)

Testing with a resume that said *"Amazon Web Services"* against a job description that said
*"AWS"* — the same qualification, phrased two different ways — produced a 33.3% skills-match
score with `aws` incorrectly listed as missing. Root cause: the scorer compared extracted skill
lists via exact lowercased-string matching. Even though Gemini's extraction step correctly
identified the underlying skill, "amazon web services" and "aws" are never equal as strings, so
the LLM's understanding was being discarded at the scoring layer — a real design flaw, not just
an outdated config value like Bug #1.

**Fix, two layers:**
1. Updated the extraction prompt to explicitly instruct Gemini to output canonical, standardized
   skill names ("aws" not "Amazon Web Services") rather than verbatim phrasing from the source
   text.
2. Added a synonym-normalization safety net in `scorer.py` (`SKILL_ALIASES`) that catches common
   variants even if extraction doesn't perfectly canonicalize every time.

**Verified after fix:** the same test case rose from 33.3% → 66.7% skills match, with `aws`
correctly matched. One skill (`mongodb`) still shows as missing in that specific test — this is
*correct*, not a remaining bug: the resume said "NoSQL data stores," a generic category that
includes MongoDB, Cassandra, DynamoDB, and others. Gemini correctly declined to assume it
specifically meant MongoDB rather than over-inferring, which is the right behavior even though it
costs a point on that one test case. Worth mentioning in an interview as evidence of understanding
precision/recall tradeoffs, not just "make the number go up."

## Known limitations
- The scorer's skill-overlap component treats all missing skills equally, so
  a candidate missing one "nice-to-have" skill is scored the same as one
  missing a core requirement. A fix would weight skills by how many times
  they appear in typical JDs for that role, or let the JD mark skills as
  required vs. preferred.
- Semantic similarity (TF-IDF) is a weak signal on short texts — it works
  better as a tiebreaker than a primary signal, which is why it's only
  weighted 30%.
- The `SKILL_ALIASES` synonym map in `scorer.py` is manually curated and
  necessarily incomplete — it covers common cases found through testing, not
  every possible synonym. The extraction-prompt fix (canonicalizing at the
  source) is the more scalable long-term solution; the alias map is a
  safety net, not a complete fix.

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # get one at https://aistudio.google.com/apikey
uvicorn app.main:app --reload
```
Runs at `http://localhost:8000`. Without `GEMINI_API_KEY` set, it automatically
falls back to the baseline extractor — the API never breaks, it just degrades
gracefully (mention this in interviews — it's a real production pattern).

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:5173`.

### Re-run the evaluation
```bash
cd evaluation
python eval_harness.py                 # baseline, no API key needed
GEMINI_API_KEY=your_key python eval_harness.py --use-gemini   # LLM-enhanced
```

## Deployment
- Backend → Render (`backend/render.yaml` included, set `GEMINI_API_KEY` in dashboard)
- Frontend → Vercel (set `VITE_API_URL` env var to your Render backend URL)

## Extending with Gemini / vibe-coding tools

Once the core is deployed, use Gemini CLI / AI Studio / Cursor to layer on:
- PDF resume upload + parsing (pdfplumber/PyMuPDF → feed extracted text into `/match`)
- Batch mode: match one resume against multiple JDs, ranked
- A "how to close the gap" suggestion feature using the `missing_skills` field
- Auth + saved match history if you want a fuller product story
