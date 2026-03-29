/**
 * Structured logging for the MindSim viewer.
 *
 * Provides tagged log functions, a ring buffer accessible from devtools,
 * timed fetch wrapper, and periodic flush to the server log file.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LogEntry {
  ts: string; // ISO 8601
  level: 'info' | 'warn' | 'error';
  tag: string; // e.g. "fetch", "cad-steps", "assembly"
  msg: string;
  data?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Extract a human-readable message from an unknown thrown value. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ---------------------------------------------------------------------------
// Ring buffer (circular — no shift/unshift)
// ---------------------------------------------------------------------------

const LOG_BUFFER_SIZE = 200;
const _ring: (LogEntry | undefined)[] = new Array(LOG_BUFFER_SIZE);
let _head = 0; // next write index
let _count = 0;

function _push(entry: LogEntry): void {
  _ring[_head] = entry;
  _head = (_head + 1) % LOG_BUFFER_SIZE;
  if (_count < LOG_BUFFER_SIZE) _count++;
}

/** Snapshot the ring buffer contents in chronological order. */
function _snapshot(): LogEntry[] {
  if (_count === 0) return [];
  const start = (_head - _count + LOG_BUFFER_SIZE) % LOG_BUFFER_SIZE;
  const out: LogEntry[] = [];
  for (let i = 0; i < _count; i++) {
    out.push(_ring[(start + i) % LOG_BUFFER_SIZE]!);
  }
  return out;
}

// Expose a live view on window for devtools access.
// Getter rebuilds from ring on each access — devtools usage is infrequent.
Object.defineProperty(window, '__mindsim_logs', { get: _snapshot });

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

const _consoleMethods = { info: 'log', warn: 'warn', error: 'error' } as const;

function _log(level: 'info' | 'warn' | 'error', tag: string, msg: string, data?: Record<string, unknown>): void {
  const prefix = `[${tag}]`;
  const method = _consoleMethods[level];
  if (data) {
    console[method](prefix, msg, data);
  } else {
    console[method](prefix, msg);
  }
  _push({ ts: new Date().toISOString(), level, tag, msg, data });
}

// ---------------------------------------------------------------------------
// Public log functions
// ---------------------------------------------------------------------------

export function info(tag: string, msg: string, data?: Record<string, unknown>): void {
  _log('info', tag, msg, data);
}

export function warn(tag: string, msg: string, data?: Record<string, unknown>): void {
  _log('warn', tag, msg, data);
}

export function error(tag: string, msg: string, data?: Record<string, unknown>): void {
  _log('error', tag, msg, data);
}

// ---------------------------------------------------------------------------
// Timed fetch wrapper
// ---------------------------------------------------------------------------

/** Wraps native fetch with timing and structured logging. */
export async function timedFetch(url: string, init?: RequestInit): Promise<Response> {
  const method = init?.method ?? 'GET';
  const t0 = performance.now();
  try {
    const resp = await fetch(url, init);
    const durationMs = Math.round(performance.now() - t0);
    info('fetch', `${method} ${url} -> ${resp.status} (${durationMs}ms)`, {
      method,
      url,
      status: resp.status,
      duration_ms: durationMs,
    });
    return resp;
  } catch (err) {
    const durationMs = Math.round(performance.now() - t0);
    error('fetch', `${method} ${url} -> ERROR (${durationMs}ms)`, {
      method,
      url,
      duration_ms: durationMs,
      error: errorMessage(err),
    });
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Server flush
// ---------------------------------------------------------------------------

let _flushing = false;
let _consecutiveFailures = 0;
const MAX_FLUSH_FAILURES = 5;

/** POST buffered log entries to the server and clear the buffer. Fire-and-forget. */
export async function flushToServer(): Promise<void> {
  const entries = _snapshot();
  if (entries.length === 0 || _flushing) return;
  if (_consecutiveFailures >= MAX_FLUSH_FAILURES) return; // back off permanently until a page reload

  _flushing = true;
  try {
    const resp = await fetch('/api/client-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entries }),
    });
    if (resp.ok) {
      // Clear only the entries we sent (new ones may have arrived during POST)
      _count = 0;
      _head = 0;
      _consecutiveFailures = 0;
    } else {
      _consecutiveFailures++;
    }
  } catch {
    _consecutiveFailures++;
  } finally {
    _flushing = false;
  }
}

// Periodic flush every 30 seconds
setInterval(flushToServer, 30_000);

// Flush on unhandled errors
window.addEventListener('error', () => {
  flushToServer();
});
window.addEventListener('unhandledrejection', () => {
  flushToServer();
});
