import { useState, useCallback, useEffect } from 'react'
import CodeEditor from './components/CodeEditor'
import ReviewPanel from './components/ReviewPanel'
import RefactorPanel from './components/RefactorPanel'
import ScoreCard from './components/ScoreCard'
import { reviewCode, refactorCode, checkHealth } from './api/client'
import styles from './App.module.css'

const TABS = ['Review', 'Refactor']

const EXAMPLE_CODE = `def find_duplicates(nums):
    duplicates = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                if nums[i] not in duplicates:
                    duplicates.append(nums[i])
    return duplicates

def calculate_stats(data):
    total = 0
    count = 0
    for item in data:
        total = total + item
        count = count + 1
    average = total / count
    unused_var = 42
    return average
`

export default function App() {
  const [code, setCode] = useState(EXAMPLE_CODE)
  const [activeTab, setActiveTab] = useState('Review')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [reviewResult, setReviewResult] = useState(null)
  const [refactorResult, setRefactorResult] = useState(null)
  const [level, setLevel] = useState('medium')
  const [useAi, setUseAi] = useState(true)
  // Off by default -- experimental, opt-in per the plan (34% differential-
  // verification pass rate as of introduction). codet5Available starts
  // false and is confirmed via /health on mount, not assumed true, so the
  // checkbox is never offered as usable before we actually know it is --
  // same "disabled with a tooltip when unavailable, not silently missing"
  // requirement the plan states explicitly for this specific toggle.
  const [useCodet5, setUseCodet5] = useState(false)
  const [codet5Available, setCodet5Available] = useState(false)

  useEffect(() => {
    checkHealth()
      .then(health => setCodet5Available(Boolean(health?.codet5_available)))
      .catch(() => setCodet5Available(false)) // backend unreachable -- stay disabled, not an error state here
  }, [])

  const handleReview = useCallback(async () => {
    if (!code.trim()) return
    setLoading(true)
    setError(null)
    setReviewResult(null)
    try {
      const result = await reviewCode(code)
      setReviewResult(result)
      setActiveTab('Review')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [code])

  const handleRefactor = useCallback(async () => {
    if (!code.trim()) return
    setLoading(true)
    setError(null)
    setRefactorResult(null)
    try {
      const result = await refactorCode(code, level, useAi, useCodet5 && codet5Available)
      setRefactorResult(result)
      setActiveTab('Refactor')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [code, level, useAi, useCodet5, codet5Available])

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⚡</span>
          <span className={styles.logoText}>ACRR</span>
          <span className={styles.logoSub}>Automated Code Review & Refactoring</span>
        </div>
        <div className={styles.headerControls}>
          <label className={styles.controlLabel}>
            Level
            <select
              className={styles.select}
              value={level}
              onChange={e => setLevel(e.target.value)}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </label>
          <label className={styles.controlLabel}>
            <input
              type="checkbox"
              checked={useAi}
              onChange={e => setUseAi(e.target.checked)}
              className={styles.checkbox}
            />
            Use AI
          </label>
          <label
            className={styles.controlLabel}
            title={
              codet5Available
                ? 'Experimental local model, runs on our server (not your browser) — usually 5–15 seconds. ' +
                  'Every result is behaviourally verified before being shown; the model is right about ' +
                  '1 in 3 times, so "no suggestion" is a normal, expected outcome, not an error.'
                : 'CodeT5+ model not installed on the backend — this is a local, opt-in feature ' +
                  'that requires the fine-tuned model files to be present.'
            }
          >
            <input
              type="checkbox"
              checked={useCodet5}
              disabled={!codet5Available}
              onChange={e => setUseCodet5(e.target.checked)}
              className={styles.checkbox}
            />
            CodeT5+ (local, slower)
          </label>
          <button
            className={`${styles.btn} ${styles.btnSecondary}`}
            onClick={handleReview}
            disabled={loading}
          >
            {loading && activeTab === 'Review' ? '⏳ Reviewing…' : '🔍 Review'}
          </button>
          <button
            className={`${styles.btn} ${styles.btnPrimary}`}
            onClick={handleRefactor}
            disabled={loading}
          >
            {loading && activeTab === 'Refactor' ? '⏳ Refactoring…' : '✨ Refactor'}
          </button>
        </div>
      </header>

      {/* Main layout */}
      <main className={styles.main}>
        {/* Left pane — code editor */}
        <section className={styles.editorPane}>
          <div className={styles.paneHeader}>
            <span className={styles.paneTitle}>📝 Your Code</span>
            <span className={styles.hint}>Python only</span>
          </div>
          <CodeEditor value={code} onChange={setCode} />
        </section>

        {/* Right pane — results */}
        <section className={styles.resultsPane}>
          {/* Score card (shown when review result available) */}
          {reviewResult?.quality_score && (
            <ScoreCard score={reviewResult.quality_score} />
          )}

          {/* Error banner */}
          {error && (
            <div className={styles.errorBanner}>
              ❌ {error}
            </div>
          )}

          {/* Tabs */}
          <div className={styles.tabs}>
            {TABS.map(tab => (
              <button
                key={tab}
                className={`${styles.tab} ${activeTab === tab ? styles.tabActive : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab}
                {tab === 'Review' && reviewResult && (
                  <span className={styles.badge}>
                    {reviewResult.issues?.length ?? 0}
                  </span>
                )}
                {tab === 'Refactor' && refactorResult && (
                  <span className={styles.badge}>
                    {refactorResult.applied_rules?.length ?? 0}
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className={styles.tabContent}>
            {activeTab === 'Review' && (
              <ReviewPanel result={reviewResult} loading={loading} />
            )}
            {activeTab === 'Refactor' && (
              <RefactorPanel result={refactorResult} loading={loading} />
            )}
          </div>
        </section>
      </main>
    </div>
  )
}
