import Editor from '@monaco-editor/react'

/**
 * CodeEditor — VS Code's Monaco editor wrapped for our app.
 *
 * Props:
 *   value    {string}    current code
 *   onChange {function}  called with new code string on change — omit for a read-only view
 *   readOnly {boolean}   disables editing (used for the output/refactored-code pane)
 */
export default function CodeEditor({ value, onChange, readOnly = false }) {
  return (
    <Editor
      height="100%"
      defaultLanguage="python"
      theme="vs-dark"
      value={value}
      onChange={v => onChange?.(v ?? '')}
      options={{
        fontSize: 13,
        fontFamily: "'Fira Code', 'Cascadia Code', Consolas, monospace",
        fontLigatures: true,
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        lineNumbers: 'on',
        renderLineHighlight: readOnly ? 'none' : 'line',
        smoothScrolling: true,
        cursorBlinking: 'smooth',
        tabSize: 4,
        insertSpaces: true,
        automaticLayout: true,
        padding: { top: 12, bottom: 12 },
        readOnly,
        domReadOnly: readOnly,
      }}
    />
  )
}
