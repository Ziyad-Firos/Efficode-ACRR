import { useState } from 'react'
import ReactDiffViewer from 'react-diff-viewer-continued'
import styles from './RefactorPanel.module.css'

/**
 * RefactorPanel — shows applied rules, diff view, and AI suggestions.
 *
 * Props:
 *   result  {object|null}  RefactorResponse from backend
 *   loading {boolean}
 */
export default function RefactorPanel({ result, loading }) {
  const [view, setView] = useState('complexity')   // 'complexity' | 'diff' | 'rules' | 'ai'

  if (loading) return <Placeholder text="Refactoring your code…" spin />
  if (!result)  return <Placeholder text="Click ✨ Refactor to optimise your code." />

  const rules       = result.applied_rules ?? []
  // applied=false entries are advice that did not modify the code
  const appliedRules    = rules.filter(r => r.applied !== false)
  const suggestedRules  = rules.filter(r => r.applied === false)
  const ruleCount = appliedRules.length
  const aiCount   = result.ai_suggestions?.filter(s => s.validated).length ?? 0
  const hasChanges = result.original_code !== result.refactored_code

  return (
    <div className={styles.panel}>
      {/* Summary bar */}
      <div className={styles.summaryBar}>
        <span className={styles.summary}>{result.summary}</span>
        {result.complexity && (
          <ComplexityBadge complexity={result.complexity} />
        )}
        {!result.ai_available && (
          <span className={styles.aiUnavailable} title="Set GEMINI_API_KEY or OLLAMA_BASE_URL to enable">
            ⚡ AI unavailable — rule-based only
          </span>
        )}
      </div>

      {/* Sub-tabs */}
      <div className={styles.subTabs}>
        {result.complexity && (
          <button
            className={`${styles.subTab} ${view === 'complexity' ? styles.subTabActive : ''}`}
            onClick={() => setView('complexity')}
          >
            ⏱️ Complexity
          </button>
        )}
        <button
          className={`${styles.subTab} ${view === 'diff' ? styles.subTabActive : ''}`}
          onClick={() => setView('diff')}
        >
          🔀 Diff {!hasChanges && <span className={styles.noChange}>(no changes)</span>}
        </button>
        <button
          className={`${styles.subTab} ${view === 'rules' ? styles.subTabActive : ''}`}
          onClick={() => setView('rules')}
        >
          📋 Applied Rules
          {ruleCount > 0 && <span className={styles.count}>{ruleCount}</span>}
        </button>
        {result.ai_suggestions?.length > 0 && (
          <button
            className={`${styles.subTab} ${view === 'ai' ? styles.subTabActive : ''}`}
            onClick={() => setView('ai')}
          >
            🤖 AI Suggestions
            {aiCount > 0 && <span className={styles.count}>{aiCount}</span>}
          </button>
        )}
      </div>

      {/* Content */}
      <div className={styles.content}>
        {view === 'complexity' && (
          result.complexity
            ? <ComplexityReport complexity={result.complexity} />
            : <p className={styles.empty}>
                Complexity prediction unavailable — scikit-learn is not installed
                on the backend.
              </p>
        )}

        {view === 'diff' && (
          hasChanges ? (
            <div className={styles.diffWrap}>
              <ReactDiffViewer
                oldValue={result.original_code}
                newValue={result.refactored_code}
                splitView={false}
                useDarkTheme
                hideLineNumbers={false}
                showDiffOnly={false}
                styles={{
                  variables: {
                    dark: {
                      diffViewerBackground: '#0f1117',
                      addedBackground: 'rgba(85, 239, 196, 0.08)',
                      addedColor: '#55efc4',
                      removedBackground: 'rgba(255, 107, 107, 0.08)',
                      removedColor: '#ff6b6b',
                      wordAddedBackground: 'rgba(85, 239, 196, 0.2)',
                      wordRemovedBackground: 'rgba(255, 107, 107, 0.2)',
                      codeFoldBackground: '#1a1d27',
                      emptyLineBackground: '#0f1117',
                      gutterBackground: '#1a1d27',
                      gutterColor: '#8892b0',
                    },
                  },
                  line: { fontFamily: "'Fira Code', Consolas, monospace", fontSize: '12px' },
                }}
              />
            </div>
          ) : (
            <div className={styles.noChanges}>
              ✅ Code is already optimised at this level. Try a higher level.
            </div>
          )
        )}

        {view === 'rules' && (
          <div className={styles.rulesList}>
            {ruleCount === 0 && suggestedRules.length === 0 && (
              <p className={styles.empty}>No rules were applied.</p>
            )}

            {appliedRules.map((rule, i) => (
              <div key={`a${i}`} className={styles.ruleRow}>
                <div className={styles.ruleMeta}>
                  <span className={styles.ruleTag}>{rule.rule}</span>
                  {rule.line != null && (
                    <span className={styles.lineNum}>Line {rule.line}</span>
                  )}
                  <span className={styles.appliedBadge}>changed</span>
                </div>
                <p className={styles.ruleDesc}>{rule.description}</p>
              </div>
            ))}

            {suggestedRules.length > 0 && (
              <div className={styles.adviceSection}>
                <h4 className={styles.adviceHeading}>
                  Suggestions — not applied automatically
                </h4>
                <p className={styles.adviceIntro}>
                  These changes could alter behaviour depending on your data types,
                  so they are reported rather than applied.
                </p>
                {suggestedRules.map((rule, i) => (
                  <div key={`s${i}`} className={`${styles.ruleRow} ${styles.ruleRowAdvice}`}>
                    <div className={styles.ruleMeta}>
                      <span className={styles.ruleTag}>{rule.rule}</span>
                      {rule.line != null && (
                        <span className={styles.lineNum}>Line {rule.line}</span>
                      )}
                      <span className={styles.adviceBadge}>advice</span>
                    </div>
                    <p className={styles.ruleDesc}>{rule.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {view === 'ai' && (
          <div className={styles.aiList}>
            {result.ai_suggestions.map((s, i) => (
              <div key={i} className={`${styles.aiCard} ${!s.validated ? styles.aiInvalid : ''}`}>
                <div className={styles.aiHeader}>
                  <span className={styles.aiNum}>Suggestion {i + 1}</span>
                  {s.validated
                    ? <span className={styles.validated}>✅ Valid Python</span>
                    : <span className={styles.invalid}>⚠️ Could not validate</span>
                  }
                </div>
                <p className={styles.aiExplanation}>{s.explanation}</p>
                {s.diff && (
                  <pre className={styles.aiDiff}>{s.diff}</pre>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function ComplexityReport({ complexity }) {
  const improved = complexity.before !== complexity.after
  const confidence = Math.round(complexity.confidence * 100)
  const confLevel = confidence >= 80 ? 'high' : confidence >= 60 ? 'medium' : 'low'

  return (
    <div className={styles.complexityReport}>
      {/* Both sides are always shown, even when identical — seeing that the
          complexity did NOT change is as informative as seeing that it did. */}
      <div className={styles.bigORow}>
        <div className={styles.bigOBlock}>
          <span className={styles.bigOLabel}>Your code</span>
          <span className={styles.bigOValue}>{complexity.before}</span>
        </div>

        <span className={styles.bigOArrow}>→</span>

        <div className={styles.bigOBlock}>
          <span className={styles.bigOLabel}>After refactor</span>
          <span className={`${styles.bigOValue} ${improved ? styles.bigOImproved : styles.bigOSame}`}>
            {complexity.after}
          </span>
        </div>

        {improved
          ? <span className={styles.deltaImproved}>↓ improved</span>
          : <span className={styles.deltaSame}>unchanged</span>}
      </div>

      {!improved && (
        <p className={styles.unchangedNote}>
          The refactor tidied the code without changing how it scales. Rule-based
          transforms remove redundancy — they do not redesign the algorithm. Use
          the suggestion below for that.
        </p>
      )}

      <div className={`${styles.confRow} ${styles['conf_' + confLevel]}`}>
        <div className={styles.confBarTrack}>
          <div className={styles.confBarFill} style={{ width: `${confidence}%` }} />
        </div>
        <span className={styles.confText}>{confidence}% confidence</span>
      </div>

      {complexity.explanation?.length > 0 && (
        <div className={styles.reasonBlock}>
          <h4 className={styles.reasonTitle}>Why</h4>
          <ul className={styles.reasonList}>
            {complexity.explanation.map((line, i) => (
              <li key={i} className={styles.reasonItem}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {complexity.functions?.length > 1 && (
        <div className={styles.perFunctionBlock}>
          <h4 className={styles.reasonTitle}>Per function</h4>
          <div className={styles.functionList}>
            {complexity.functions.map((fn, i) => (
              <div
                key={i}
                className={`${styles.functionRow} ${fn.is_dominant ? styles.functionDominant : ''}`}
              >
                <div className={styles.functionHead}>
                  <span className={styles.functionName}>{fn.name}</span>
                  {fn.line != null && (
                    <span className={styles.lineNum}>Line {fn.line}</span>
                  )}
                  <span className={styles.functionBigO}>{fn.complexity}</span>
                  <span className={styles.functionConf}>
                    {Math.round(fn.confidence * 100)}%
                  </span>
                  {fn.is_dominant && (
                    <span className={styles.dominantBadge}>slowest</span>
                  )}
                </div>
                {fn.explanation?.length > 0 && (
                  <ul className={styles.functionReasons}>
                    {fn.explanation.map((line, j) => (
                      <li key={j} className={styles.functionReason}>{line}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {complexity.suggestion && (
        <div className={styles.suggestionBlock}>
          <h4 className={styles.suggestionTitle}>💡 How to make it faster</h4>
          <p className={styles.suggestionText}>{complexity.suggestion}</p>
        </div>
      )}
    </div>
  )
}

function ComplexityBadge({ complexity }) {
  const improved = complexity.before !== complexity.after
  return (
    <span className={`${styles.complexityBadge} ${improved ? styles.complexityImproved : ''}`}>
      {improved
        ? `${complexity.before} → ${complexity.after}`
        : complexity.after
      }
      {' '}
      <span className={styles.confidence}>({Math.round(complexity.confidence * 100)}% conf.)</span>
    </span>
  )
}

function Placeholder({ text, spin }) {
  return (
    <div className={styles.placeholder}>
      {spin && <span className={styles.spinner} />}
      <p>{text}</p>
    </div>
  )
}
