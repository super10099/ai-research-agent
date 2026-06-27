import { useState, useRef, useCallback, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

// ── Phase constants ───────────────────────────────────────────────────────────
const PHASE = {
  IDLE:        'idle',
  RESEARCHING: 'researching',   // POST /api/research in-flight (~20-40s)
  AWAITING:    'awaiting',      // Research done; waiting for human approval
  SYNTHESIZING:'synthesizing',  // EventSource open; tokens streaming in
  COMPLETE:    'complete',      // Report fully received
  ERROR:       'error',
}

const PHASE_LABEL = {
  [PHASE.RESEARCHING]:  'Running planner → researcher → critic…',
  [PHASE.AWAITING]:     'Research complete. Review below and approve synthesis.',
  [PHASE.SYNTHESIZING]: 'Synthesizing report…',
  [PHASE.COMPLETE]:     'Report complete.',
  [PHASE.ERROR]:        'An error occurred.',
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [phase,   setPhase]   = useState(PHASE.IDLE)
  const [topic,   setTopic]   = useState('')
  const [session, setSession] = useState(null)   // ResearchSessionResponse from API
  const [report,  setReport]  = useState('')     // accumulated streaming text
  const [error,   setError]   = useState(null)

  // Hold the EventSource in a ref so we can close it on unmount or reset.
  const esRef = useRef(null)

  // ── Phase 1: run planner + researcher + critic ────────────────────────────
  const startResearch = async () => {
    if (!topic.trim() || phase === PHASE.RESEARCHING) return

    setPhase(PHASE.RESEARCHING)
    setSession(null)
    setReport('')
    setError(null)

    try {
      const res = await fetch('/api/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: topic.trim() }),
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setSession(data)
      setPhase(PHASE.AWAITING)
    } catch (err) {
      setError(err.message)
      setPhase(PHASE.ERROR)
    }
  }

  // ── Phase 2: open SSE stream to synthesizer ───────────────────────────────
  const startSynthesis = useCallback(() => {
    if (!session) return

    setPhase(PHASE.SYNTHESIZING)
    setReport('')

    /*
     * EventSource — the browser's built-in SSE client.
     *
     * How it works under the hood:
     * 1. Opens a GET request with Accept: text/event-stream.
     * 2. Keeps the TCP connection open indefinitely.
     * 3. Parses the response body line-by-line:
     *      "data: {...}\n\n"  →  fires an onmessage event
     *      ":\n\n"            →  server keepalive ping, silently ignored
     * 4. If the connection drops, the browser automatically reopens it
     *    after ~3 seconds (built-in reconnect — no application code needed).
     *    It sends "Last-Event-ID: <id>" if the server set ids on events.
     *
     * This differs from fetch() streaming (ReadableStream) in that EventSource
     * handles reconnect, parses the event-stream format, and exposes a clean
     * event-listener API rather than a byte-level reader.
     */
    const es = new EventSource(`/api/research/${session.session_id}/stream`)
    esRef.current = es

    es.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'token') {
        /*
         * Functional updater: setReport(prev => prev + token)
         *
         * React 18 Automatic Batching means multiple state updates within
         * the same event loop tick are batched into one re-render.  EventSource
         * fires onmessage callbacks as microtasks — if tokens arrive fast enough
         * (>60/s), React may batch several of them.  The functional form ensures
         * each update sees the latest accumulated text, not a stale closure value.
         */
        setReport(prev => prev + data.content)
      } else if (data.type === 'done') {
        setPhase(PHASE.COMPLETE)
        es.close()
      } else if (data.type === 'error') {
        setError(data.message)
        setPhase(PHASE.ERROR)
        es.close()
      }
    }

    es.onerror = () => {
      // onerror fires on both transient disconnects (browser will retry) and
      // fatal errors.  We treat it as fatal here for simplicity.
      if (phase !== PHASE.COMPLETE) {
        setError('Connection to server lost. Check that the backend is running.')
        setPhase(PHASE.ERROR)
      }
      es.close()
    }
  }, [session, phase])

  // ── Reset ─────────────────────────────────────────────────────────────────
  const reset = () => {
    esRef.current?.close()
    esRef.current = null
    setPhase(PHASE.IDLE)
    setSession(null)
    setReport('')
    setError(null)
  }

  // Close EventSource on unmount to prevent memory leaks.
  useEffect(() => () => esRef.current?.close(), [])

  const isActive = phase === PHASE.RESEARCHING || phase === PHASE.SYNTHESIZING

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app">
      <header>
        <h1>AI Research Agent</h1>
        <p>Multi-agent · RAG + web search · Streaming synthesis</p>
      </header>

      {/* ── Input form ── */}
      <div className="research-form">
        <input
          type="text"
          placeholder="e.g. How does retrieval-augmented generation work?"
          value={topic}
          onChange={e => setTopic(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && startResearch()}
          disabled={isActive}
        />
        <button
          className="btn-primary"
          onClick={startResearch}
          disabled={isActive || !topic.trim()}
        >
          {phase === PHASE.RESEARCHING ? 'Researching…' : 'Research'}
        </button>
        {phase !== PHASE.IDLE && (
          <button className="btn-reset" onClick={reset} disabled={isActive}>
            Reset
          </button>
        )}
      </div>

      {/* ── Phase status bar ── */}
      {phase !== PHASE.IDLE && (
        <div className={`phase-bar phase-${phase}`}>
          <span className="dot" />
          {PHASE_LABEL[phase]}
        </div>
      )}

      {/* ── Error ── */}
      {phase === PHASE.ERROR && error && (
        <div className="error-box">Error: {error}</div>
      )}

      {/* ── Research summary (shown after research, before synthesis) ── */}
      {session && phase !== PHASE.IDLE && (
        <div className="summary-card">
          <h2>
            Research Summary
            <span className="iter-badge">{session.iteration} iteration{session.iteration !== 1 ? 's' : ''}</span>
          </h2>

          <ul className="sub-questions">
            {session.sub_questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>

          {session.critique && (
            <div className="critique-text">{session.critique}</div>
          )}

          {session.gaps?.length > 0 && (
            <div className="gaps">
              {session.gaps.map((g, i) => (
                <span key={i} className="gap-tag">{g}</span>
              ))}
            </div>
          )}

          <hr className="divider" />

          {phase === PHASE.AWAITING && (
            <button className="btn-approve" onClick={startSynthesis}>
              Approve &amp; Generate Report
            </button>
          )}
        </div>
      )}

      {/* ── Streaming / final report ── */}
      {(phase === PHASE.SYNTHESIZING || phase === PHASE.COMPLETE) && report && (
        <div className="report-section">
          <h2>Report</h2>

          {phase === PHASE.SYNTHESIZING ? (
            /*
             * During streaming: render raw text in a monospaced box.
             * We intentionally avoid parsing markdown per-token — every
             * token would cause a full ReactMarkdown re-parse, adding
             * ~1-2ms of CPU work per token and causing visible flicker
             * as headings partially form.  Plain text is stable.
             */
            <pre className="stream-box">{report}</pre>
          ) : (
            /*
             * After streaming completes: render the full markdown.
             * This is a one-time parse of the complete document.
             */
            <div className="rendered-report">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
