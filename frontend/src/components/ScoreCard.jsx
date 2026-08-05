import styles from './ScoreCard.module.css'

const GRADE_COLOR = {
  A: '#55efc4',
  B: '#74b9ff',
  C: '#ffd166',
  D: '#ff9f43',
  F: '#ff6b6b',
}

/**
 * ScoreCard — displays the overall quality score and breakdown.
 *
 * Props:
 *   score  {object}  QualityScore from backend
 *     .score      {number} 0-100
 *     .grade      {string} A-F
 *     .breakdown  {object} {style, complexity, security, maintainability}
 */
export default function ScoreCard({ score }) {
  const color = GRADE_COLOR[score.grade] ?? '#8892b0'

  return (
    <div className={styles.card}>
      {/* Overall score */}
      <div className={styles.overall}>
        <div className={styles.gradeRing} style={{ '--grade-color': color }}>
          <span className={styles.grade} style={{ color }}>{score.grade}</span>
        </div>
        <div className={styles.scoreInfo}>
          <span className={styles.scoreNum} style={{ color }}>{score.score}</span>
          <span className={styles.scoreLabel}>/100</span>
        </div>
      </div>

      {/* Breakdown bars */}
      <div className={styles.breakdown}>
        <BreakdownBar label="Style"         value={score.breakdown.style} />
        <BreakdownBar label="Complexity"    value={score.breakdown.complexity} />
        <BreakdownBar label="Security"      value={score.breakdown.security} />
        <BreakdownBar label="Maintainability" value={score.breakdown.maintainability} />
      </div>
    </div>
  )
}

function BreakdownBar({ label, value }) {
  const color = value >= 80 ? '#55efc4' : value >= 60 ? '#ffd166' : '#ff6b6b'
  return (
    <div className={styles.barRow}>
      <span className={styles.barLabel}>{label}</span>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{ width: `${value}%`, background: color }}
        />
      </div>
      <span className={styles.barValue} style={{ color }}>{value}</span>
    </div>
  )
}
