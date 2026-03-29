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
// Ring buffer
// ---------------------------------------------------------------------------

const LOG_BUFFER_SIZE = 200;
const _buffer: LogEntry[] = [];

function _push(entry: LogEntry): void {
  _buffer.push(entry);
  if (_buffer.length > LOG_BUFFER_SIZE) {
    _buffer.shift();
  }
}

// Expose buffer on window for devtools access
(window as any).__mindsim_logs = _buffer;

// ---------------------------------------------------------------------------
// Console output
// ---------------------------------------------------------------------------

function _consoleLog(level: 'log' | 'warn' | 'error', tag: string, msg: string, data?: Record<string, unknown>): void {
  const prefix = `[${tag}]`;
  if (data) {
    console[level](prefix, msg, data);
  } else {
    console[level](prefix, msg);
  }
}

// ---------------------------------------------------------------------------
// Public log functions
// ---------------------------------------------------------------------------

export function info(tag: string, msg: string, data?: Record<string, unknown>): void {
  _consoleLog('log', tag, msg, data);
  _push({ ts: new Date().toISOString(), level: 'info', tag, msg, data });
}

export function warn(tag: string, msg: string, data?: Record<string, unknown>): void {
  _consoleLog('warn', tag, msg, data);
  _push({ ts: new Date().toISOString(), level: 'warn', tag, msg, data });
}

export function error(tag: string, msg: string, data?: Record<string, unknown>): void {
  _consoleLog('error', tag, msg, data);
  _push({ ts: new Date().toISOString(), level: 'error', tag, msg, data });
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
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Server flush
// ---------------------------------------------------------------------------

let _flushing = false;

/** POST buffered log entries to the server and clear the buffer. Fire-and-forget. */
export async function flushToServer(): Promise<void> {
  if (_buffer.length === 0 || _flushing) return;
  _flushing = true;
  const entries = _buffer.splice(0, _buffer.length);
  try {
    const resp = await fetch('/api/client-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ entries }),
    });
    if (!resp.ok) {
      // Re-queue entries at the front
      _buffer.unshift(...entries);
    }
  } catch {
    // Re-queue entries — server may not be available
    _buffer.unshift(...entries);
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
