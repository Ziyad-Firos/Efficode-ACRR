/**
 * api/client.js — Single source of truth for all API calls.
 *
 * ONE file, ONE base URL. No port mismatches possible.
 * During development, Vite proxies /review and /refactor to localhost:8000.
 * In production, set VITE_API_BASE_URL to the deployed backend URL.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

async function _post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  const data = await res.json()

  if (!res.ok) {
    throw new Error(data.error ?? `Request failed with status ${res.status}`)
  }

  return data
}

/**
 * Review code — returns ReviewResponse shape.
 * @param {string} code
 */
export async function reviewCode(code) {
  return _post('/review', { code })
}

/**
 * Refactor code — returns RefactorResponse shape.
 * @param {string} code
 * @param {'low'|'medium'|'high'} level
 * @param {boolean} useAi
 */
export async function refactorCode(code, level = 'medium', useAi = true) {
  return _post('/refactor', { code, level, use_ai: useAi })
}

/**
 * Health check.
 */
export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`)
  return res.json()
}
