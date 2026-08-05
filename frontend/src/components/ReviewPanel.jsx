import styles from './ReviewPanel.module.css'

const SEVERITY_ICON = {
  error:   '🔴',
  warning: '🟡',
  info:    '🔵',
  style:   '⚪',
}

const SEVERITY_ORDER = { error: 0, warning: 1, info: 2, style: 3 }

const CATEGORY_LABEL = {
  style:      '✏️ Style',
  complexity: '📊 Complexity',
  security:   '🔒 Security',
  smell:      '🐛 Code Smell',
  syntax:     '❌ Syntax',
}

/**
 * ReviewPanel — displays review issues grouped by category.
 *
 * Props:
 *   result  {object|null}  ReviewResponse from backend
 *   loading {boolean}
 */
export default function ReviewPanel({ result, loading }) {
  if (loading) return <Placeholder text="Analysing your code…" spin />
  if (!result)  return <Placeholder text="Click 🔍 Review to analyse your code." />

  if (!result.valid) {
    return (
      <div className={styles.syntaxError}>
        <span className={styles.syntaxIcon}>❌</span>
        <div>
          <strong>Syntax Error</strong>
          <p>{result.syntax_error}</p>
        </div>
      </div>
    )
  }

  if (!result.issues?.length) {
    return (
      <div className={styles.clean}>
        <span>✅</span>
        <strong>No issues found!</strong>
        <p>{result.summary}</p>
      </div>
    )
  }

  // Group issues by category
  const grouped = {}
  for (const issue of result.issues) {
    const cat = issue.category
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(issue)
  }

  // Sort issues within each group by severity
  for (const cat of Object.keys(grouped)) {
    grouped[cat].sort((a, b) =>
      (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
    )
  }

  const categoryOrder = ['security', 'complexity', 'smell', 'style', 'syntax']

  return (
    <div className={styles.panel}>
      <p className={styles.summary}>{result.summary}</p>
      {categoryOrder
        .filter(cat => grouped[cat])
        .map(cat => (
          <div key={cat} className={styles.group}>
            <div className={styles.groupHeader}>
              {CATEGORY_LABEL[cat] ?? cat}
              <span className={styles.groupCount}>{grouped[cat].length}</span>
            </div>
            {grouped[cat].map((issue, i) => (
              <IssueRow key={i} issue={issue} />
            ))}
          </div>
        ))}
    </div>
  )
}

function IssueRow({ issue }) {
  return (
    <div className={`${styles.issueRow} ${styles[`sev_${issue.severity}`]}`}>
      <span className={styles.sevIcon}>{SEVERITY_ICON[issue.severity] ?? '⚪'}</span>
      <div className={styles.issueBody}>
        <div className={styles.issueMeta}>
          {issue.line != null && (
            <span className={styles.lineNum}>Line {issue.line}</span>
          )}
          <span className={styles.ruleId}>{issue.rule}</span>
        </div>
        <p className={styles.issueMsg}>{issue.message}</p>
      </div>
    </div>
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
