/**
 * URL-based view state — encode/decode viewer state in URL query params.
 *
 * Uses history.replaceState (no new history entries). Strips defaults
 * (tab=design, mode=explore) to keep URLs clean. Preserves existing
 * routing params (bot, component, cadsteps, from).
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ViewStateParams {
  tab?: 'design' | 'sim';
  mode?: string;
  select?: string;
  solo?: string;
  step?: number;
}

/**
 * Note: routing params (bot, component, cadsteps, from) are managed by
 * the top-level router in viewer.ts — view-state never touches them.
 */

/** Default values — omitted from URL when they match. */
const DEFAULTS: Record<string, string> = {
  tab: 'design',
  mode: 'explore',
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Parse current URL into typed view-state params. */
export function readViewState(): ViewStateParams {
  const url = new URLSearchParams(window.location.search);
  const result: ViewStateParams = {};

  const tab = url.get('tab');
  if (tab === 'design' || tab === 'sim') result.tab = tab;

  const mode = url.get('mode');
  if (mode) result.mode = mode;

  const select = url.get('select');
  if (select) result.select = select;

  const solo = url.get('solo');
  if (solo) result.solo = solo;

  const step = url.get('step');
  if (step !== null) {
    const n = parseInt(step, 10);
    if (!Number.isNaN(n) && n >= 0) result.step = n;
  }

  return result;
}

/** Merge partial state into URL via replaceState. Strips default values. */
export function updateViewState(partial: Partial<ViewStateParams>): void {
  const url = new URLSearchParams(window.location.search);

  for (const [key, value] of Object.entries(partial)) {
    if (value === undefined || value === null) {
      url.delete(key);
      continue;
    }
    const strVal = String(value);
    if (DEFAULTS[key] === strVal) {
      url.delete(key);
    } else {
      url.set(key, strVal);
    }
  }

  _replaceUrl(url);
}

/** Remove specified keys from URL. */
export function clearViewState(...keys: (keyof ViewStateParams)[]): void {
  const url = new URLSearchParams(window.location.search);
  for (const key of keys) {
    url.delete(key);
  }
  _replaceUrl(url);
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function _replaceUrl(params: URLSearchParams): void {
  const qs = params.toString();
  const newUrl = qs ? `${window.location.pathname}?${qs}` : window.location.pathname;
  history.replaceState(null, '', newUrl);
}
