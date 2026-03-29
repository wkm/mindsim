/**
 * DFM Findings Panel — sortable, filterable table of DFM check results.
 *
 * Renders into a container element (typically #side-panel). Supports:
 * - Sortable columns (click header to toggle asc/desc)
 * - Severity filter toggles (error/warning/info)
 * - Assembly step filtering
 * - Progress indicator during analysis
 * - Click-to-select rows with callback
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface DFMFindingData {
  id: string;
  check_name: string;
  severity: 'error' | 'warning' | 'info';
  body: string;
  title: string;
  description: string;
  pos: [number, number, number];
  direction: [number, number, number] | null;
  measured: number | null;
  threshold: number | null;
  assembly_step: number;
  has_overlay: boolean;
}

export type FindingClickHandler = (finding: DFMFindingData) => void;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

type SortColumn = 'severity' | 'check_name' | 'body' | 'title' | 'measured';
type SortDir = 'asc' | 'desc';

const SEVERITY_ORDER: Record<string, number> = { error: 0, warning: 1, info: 2 };
const SEVERITY_ICON: Record<string, string> = { error: '\u{1F534}', warning: '\u{1F7E1}', info: '\u2139\uFE0F' };

// ---------------------------------------------------------------------------
// Styles (injected once)
// ---------------------------------------------------------------------------

let stylesInjected = false;

function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;

  const style = document.createElement('style');
  style.textContent = `
    /* DFM panel */
    .dfm-summary {
      font-size: 12px; color: var(--muted-fg); margin-bottom: 8px;
    }
    .dfm-summary .count-error { color: var(--destructive); font-weight: 600; }
    .dfm-summary .count-warning { color: var(--gold3); font-weight: 600; }

    /* Progress bar */
    .dfm-progress {
      margin-bottom: 12px; padding: 8px 10px;
      background: rgba(206,217,224,0.3); border-radius: var(--radius-sm);
    }
    .dfm-progress-label {
      font-size: 11px; color: var(--muted-fg); margin-bottom: 4px;
    }
    .dfm-progress-bar-bg {
      height: 4px; width: 100%; background: var(--muted); border-radius: 2px; overflow: hidden;
    }
    .dfm-progress-bar-fill {
      height: 100%; background: var(--primary); border-radius: 2px;
      transition: width 0.3s ease;
    }

    /* Filter toggles */
    .dfm-filters {
      display: flex; gap: 4px; margin-bottom: 8px;
    }
    .dfm-filter-btn {
      display: inline-flex; align-items: center; gap: 4px;
      height: 24px; padding: 0 8px; border-radius: var(--radius-sm);
      font-family: var(--font); font-size: 11px; font-weight: 500;
      cursor: pointer; border: 1px solid var(--border);
      background: var(--card); color: var(--secondary-fg);
      outline: none; transition: background 0.1s, opacity 0.1s;
    }
    .dfm-filter-btn:hover { background: var(--secondary); }
    .dfm-filter-btn.inactive { opacity: 0.4; }

    /* Table */
    .dfm-table-wrap {
      overflow-y: auto; max-height: calc(100vh - 240px);
      border: 1px solid var(--border); border-radius: var(--radius-sm);
    }
    .dfm-table {
      width: 100%; border-collapse: collapse; font-size: 11px;
    }
    .dfm-table th {
      position: sticky; top: 0; z-index: 1;
      background: var(--secondary); color: var(--muted-fg);
      font-weight: 600; text-align: left; padding: 4px 6px;
      cursor: pointer; user-select: none; white-space: nowrap;
      border-bottom: 1px solid var(--border);
    }
    .dfm-table th:hover { color: var(--foreground); }
    .dfm-table th .sort-arrow { font-size: 9px; margin-left: 2px; }
    .dfm-table td {
      padding: 4px 6px; border-bottom: 1px solid rgba(206,217,224,0.4);
      color: var(--dark5); vertical-align: top;
    }
    .dfm-table tr { cursor: pointer; transition: background 0.1s; }
    .dfm-table tbody tr:hover { background: rgba(19,124,189,0.06); }
    .dfm-table tbody tr.selected { background: rgba(19,124,189,0.12); }
    .dfm-table .col-sev { width: 24px; text-align: center; }
    .dfm-table .col-meas { font-family: var(--font-mono); font-size: 10px; white-space: nowrap; }
  `;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// DFMPanel class
// ---------------------------------------------------------------------------

export class DFMPanel {
  private container: HTMLElement;
  private onClick: FindingClickHandler;
  private findings: DFMFindingData[] = [];
  private maxStep: number = Number.POSITIVE_INFINITY;
  private sortCol: SortColumn = 'severity';
  private sortDir: SortDir = 'asc';
  private activeFilters: Set<string> = new Set(['error', 'warning', 'info']);
  private selectedId: string | null = null;

  // DOM refs for partial updates
  private progressEl: HTMLElement | null = null;
  private summaryEl: HTMLElement | null = null;
  private tableWrap: HTMLElement | null = null;

  constructor(container: HTMLElement, onClick: FindingClickHandler) {
    injectStyles();
    this.container = container;
    this.onClick = onClick;
    this._buildShell();
  }

  // ── Public API ──

  setFindings(findings: DFMFindingData[]): void {
    this.findings = findings;
    this._hideProgress();
    this._renderSummary();
    this._renderTable();
  }

  setProgress(state: string, checksComplete: number, checksTotal: number): void {
    if (!this.progressEl) return;
    this.progressEl.style.display = '';
    const pct = checksTotal > 0 ? Math.round((checksComplete / checksTotal) * 100) : 0;
    const label = this.progressEl.querySelector('.dfm-progress-label') as HTMLElement;
    const fill = this.progressEl.querySelector('.dfm-progress-bar-fill') as HTMLElement;
    if (label) label.textContent = `${state} (${checksComplete}/${checksTotal})`;
    if (fill) fill.style.width = `${pct}%`;
  }

  filterByStep(maxStep: number): void {
    this.maxStep = maxStep;
    this._renderSummary();
    this._renderTable();
  }

  dispose(): void {
    this.container.innerHTML = '';
    this.progressEl = null;
    this.summaryEl = null;
    this.tableWrap = null;
  }

  // ── Shell layout ──

  private _buildShell(): void {
    this.container.innerHTML = '';

    // Header
    const h2 = document.createElement('h2');
    h2.textContent = 'DFM Analysis';
    this.container.appendChild(h2);

    // Progress
    this.progressEl = document.createElement('div');
    this.progressEl.className = 'dfm-progress';
    this.progressEl.style.display = 'none';
    this.progressEl.innerHTML =
      '<div class="dfm-progress-label">Waiting...</div>' +
      '<div class="dfm-progress-bar-bg"><div class="dfm-progress-bar-fill" style="width:0%"></div></div>';
    this.container.appendChild(this.progressEl);

    // Summary
    this.summaryEl = document.createElement('div');
    this.summaryEl.className = 'dfm-summary';
    this.container.appendChild(this.summaryEl);

    // Filters
    const filters = document.createElement('div');
    filters.className = 'dfm-filters';
    for (const sev of ['error', 'warning', 'info'] as const) {
      const btn = document.createElement('button');
      btn.className = 'dfm-filter-btn';
      btn.dataset.severity = sev;
      btn.textContent = `${SEVERITY_ICON[sev]} ${sev}`;
      btn.addEventListener('click', () => this._toggleFilter(sev));
      filters.appendChild(btn);
    }
    this.container.appendChild(filters);

    // Table wrapper
    this.tableWrap = document.createElement('div');
    this.tableWrap.className = 'dfm-table-wrap';
    this.container.appendChild(this.tableWrap);
  }

  // ── Progress ──

  private _hideProgress(): void {
    if (this.progressEl) this.progressEl.style.display = 'none';
  }

  // ── Summary ──

  private _renderSummary(): void {
    if (!this.summaryEl) return;
    const visible = this._visibleFindings();
    const errors = visible.filter((f) => f.severity === 'error').length;
    const warnings = visible.filter((f) => f.severity === 'warning').length;
    const parts: string[] = [];
    if (errors > 0) parts.push(`<span class="count-error">${errors} error${errors !== 1 ? 's' : ''}</span>`);
    if (warnings > 0) parts.push(`<span class="count-warning">${warnings} warning${warnings !== 1 ? 's' : ''}</span>`);
    if (parts.length === 0) parts.push('No issues found');
    this.summaryEl.innerHTML = parts.join(', ');
  }

  // ── Filters ──

  private _toggleFilter(severity: string): void {
    if (this.activeFilters.has(severity)) {
      this.activeFilters.delete(severity);
    } else {
      this.activeFilters.add(severity);
    }
    // Update button visuals
    const btns = this.container.querySelectorAll<HTMLElement>('.dfm-filter-btn');
    for (const btn of btns) {
      const sev = btn.dataset.severity;
      if (sev) {
        btn.classList.toggle('inactive', !this.activeFilters.has(sev));
      }
    }
    this._renderSummary();
    this._renderTable();
  }

  // ── Table ──

  private _visibleFindings(): DFMFindingData[] {
    return this.findings.filter((f) => this.activeFilters.has(f.severity) && f.assembly_step <= this.maxStep);
  }

  private _sortedFindings(): DFMFindingData[] {
    const list = [...this._visibleFindings()];
    const dir = this.sortDir === 'asc' ? 1 : -1;

    list.sort((a, b) => {
      switch (this.sortCol) {
        case 'severity':
          return (SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]) * dir;
        case 'check_name':
          return a.check_name.localeCompare(b.check_name) * dir;
        case 'body':
          return a.body.localeCompare(b.body) * dir;
        case 'title':
          return a.title.localeCompare(b.title) * dir;
        case 'measured': {
          const am = a.measured ?? Number.POSITIVE_INFINITY;
          const bm = b.measured ?? Number.POSITIVE_INFINITY;
          return (am - bm) * dir;
        }
        default:
          return 0;
      }
    });
    return list;
  }

  private _renderTable(): void {
    if (!this.tableWrap) return;

    const sorted = this._sortedFindings();

    // Build table
    let html = '<table class="dfm-table"><thead><tr>';
    const columns: { key: SortColumn; label: string; cls?: string }[] = [
      { key: 'severity', label: '', cls: 'col-sev' },
      { key: 'check_name', label: 'Check' },
      { key: 'body', label: 'Body' },
      { key: 'title', label: 'Issue' },
      { key: 'measured', label: 'Value', cls: 'col-meas' },
    ];

    for (const col of columns) {
      const arrow = this.sortCol === col.key ? (this.sortDir === 'asc' ? '\u25B2' : '\u25BC') : '';
      const cls = col.cls ? ` class="${col.cls}"` : '';
      html += `<th data-sort="${col.key}"${cls}>${col.label}<span class="sort-arrow">${arrow}</span></th>`;
    }
    html += '</tr></thead><tbody>';

    for (const f of sorted) {
      const icon = SEVERITY_ICON[f.severity];
      const measStr = this._formatMeasured(f);
      const selClass = f.id === this.selectedId ? ' selected' : '';
      html += `<tr data-id="${f.id}" class="${selClass}">`;
      html += `<td class="col-sev">${icon}</td>`;
      html += `<td>${this._esc(f.check_name)}</td>`;
      html += `<td>${this._esc(f.body)}</td>`;
      html += `<td>${this._esc(f.title)}</td>`;
      html += `<td class="col-meas">${measStr}</td>`;
      html += '</tr>';
    }

    if (sorted.length === 0) {
      html += '<tr><td colspan="5" style="text-align:center;color:var(--muted-fg);padding:12px;">No findings</td></tr>';
    }

    html += '</tbody></table>';
    this.tableWrap.innerHTML = html;

    // Bind header clicks for sorting
    const headers = this.tableWrap.querySelectorAll<HTMLElement>('th[data-sort]');
    for (const th of headers) {
      th.addEventListener('click', () => {
        const col = th.dataset.sort as SortColumn;
        if (this.sortCol === col) {
          this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          this.sortCol = col;
          this.sortDir = 'asc';
        }
        this._renderTable();
      });
    }

    // Bind row clicks
    const rows = this.tableWrap.querySelectorAll<HTMLElement>('tr[data-id]');
    for (const row of rows) {
      row.addEventListener('click', () => {
        const id = row.dataset.id;
        const finding = this.findings.find((f) => f.id === id);
        if (finding) {
          this.selectedId = finding.id;
          // Update selected class
          for (const r of rows) r.classList.remove('selected');
          row.classList.add('selected');
          this.onClick(finding);
        }
      });
    }
  }

  // ── Helpers ──

  private _formatMeasured(f: DFMFindingData): string {
    if (f.measured == null) return '\u2014';
    const mStr = f.measured.toFixed(1);
    if (f.threshold != null) {
      return `${mStr} / ${f.threshold.toFixed(1)}`;
    }
    return mStr;
  }

  private _esc(s: string): string {
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  }
}
