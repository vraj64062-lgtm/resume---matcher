import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function ScoreBar({ label, value }) {
  const color = value >= 75 ? '#22c55e' : value >= 50 ? '#eab308' : '#ef4444'
  return (
    <div className="score-bar">
      <div className="score-bar-label">
        <span>{label}</span>
        <span>{value.toFixed(1)}%</span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${value}%`, background: color }} />
      </div>
    </div>
  )
}

export default function App() {
  const [resumeText, setResumeText] = useState('')
  const [jdText, setJdText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleMatch() {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resume_text: resumeText, job_description: jdText }),
      })
      if (!res.ok) throw new Error((await res.json()).detail || 'Request failed')
      setResult(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>AI Resume-to-Job Matcher</h1>
        <p>Explainable resume/JD scoring powered by Gemini + weighted matching</p>
      </header>

      <main>
        <div className="input-grid">
          <div className="input-col">
            <label>Resume</label>
            <textarea
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
              placeholder="Paste resume text here..."
              rows={12}
            />
          </div>
          <div className="input-col">
            <label>Job Description</label>
            <textarea
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              placeholder="Paste job description here..."
              rows={12}
            />
          </div>
        </div>

        <button onClick={handleMatch} disabled={loading || !resumeText || !jdText}>
          {loading ? 'Analyzing...' : 'Run Match'}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="results">
            <div className="verdict-card">
              <div className="verdict-score">{result.overall_score.toFixed(1)}%</div>
              <div className={`verdict-label verdict-${result.verdict.split(' ')[0].toLowerCase()}`}>
                {result.verdict}
              </div>
            </div>

            <ScoreBar label="Skills Match" value={result.skills_score} />
            <ScoreBar label="Experience Alignment" value={result.experience_score} />
            <ScoreBar label="Semantic Similarity" value={result.semantic_score} />

            <p className="explanation">{result.explanation}</p>

            <div className="skills-grid">
              <div>
                <h3>Matched Skills</h3>
                <div className="tags">
                  {result.skill_gap.matched_skills.map((s) => (
                    <span key={s} className="tag tag-match">{s}</span>
                  ))}
                </div>
              </div>
              <div>
                <h3>Missing Skills</h3>
                <div className="tags">
                  {result.skill_gap.missing_skills.map((s) => (
                    <span key={s} className="tag tag-missing">{s}</span>
                  ))}
                </div>
              </div>
            </div>

            <div className="latency">Processed in {result.latency_ms.toFixed(1)}ms</div>
          </div>
        )}
      </main>
    </div>
  )
}
