/**
 * DFM Mode — Design for Manufacturing analysis viewer.
 *
 * Orchestrates the DFM findings panel (left sidebar) and assembly step
 * scrubber (bottom bar). Fetches assembly sequence and DFM analysis
 * results from the API, wires finding clicks to camera fly-to + body
 * isolation, and filters findings by scrubber step.
 */

import { type AssemblyOpData, AssemblyScrubber } from './assembly-scrubber.ts';
import { type DFMFindingData, DFMPanel } from './dfm-panel.ts';
import { FocusController } from './focus-controller.ts';
import type { ViewerContext, ViewerMode } from './types.ts';

export class DFMMode implements ViewerMode {
  private ctx: ViewerContext;
  private panel: DFMPanel | null = null;
  private scrubber: AssemblyScrubber | null = null;
  private focus: FocusController;
  private pollTimer: number | null = null;
  private runId: string | null = null;
  private findings: DFMFindingData[] = [];

  /** Body name -> body ID lookup, built once on activate. */
  private bodyNameToId: Record<string, number> = {};

  constructor(ctx: ViewerContext) {
    this.ctx = ctx;
    this.focus = new FocusController(ctx);
  }

  activate(): void {
    // Build body name lookup
    this.bodyNameToId = {};
    for (const body of this.ctx.botScene.bodies) {
      this.bodyNameToId[body.name] = body.id;
    }

    // Show side panel with DFM findings
    const sidePanel = document.getElementById('side-panel');
    if (sidePanel) {
      sidePanel.style.display = '';
      sidePanel.innerHTML = '';
      this.panel = new DFMPanel(sidePanel, (finding) => this._onFindingClick(finding));
    }

    // Show scrubber in canvas container
    const container = document.getElementById('canvas-container');
    if (container) {
      this.scrubber = new AssemblyScrubber(container, (step) => this._onStepChange(step));
    }

    // Kick off data loading
    this._loadAssemblySequence();
    this._startDFMAnalysis();
  }

  deactivate(): void {
    // Stop polling
    this._stopPolling();

    // Dispose panel
    if (this.panel) {
      this.panel.dispose();
      this.panel = null;
    }
    const sidePanel = document.getElementById('side-panel');
    if (sidePanel) sidePanel.innerHTML = '';

    // Dispose scrubber
    if (this.scrubber) {
      this.scrubber.dispose();
      this.scrubber = null;
    }

    // Restore visibility
    this.ctx.botScene.showAll();
    this.ctx.syncScene();

    this.runId = null;
    this.findings = [];
  }

  update(): void {
    this.focus.update();
  }

  // ---------------------------------------------------------------------------
  // Assembly sequence
  // ---------------------------------------------------------------------------

  private async _loadAssemblySequence(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${this.ctx.botName}/assembly-sequence`);
      if (!resp.ok) {
        console.warn('[dfm] assembly-sequence fetch failed:', resp.status);
        return;
      }
      const data = await resp.json();
      const ops: AssemblyOpData[] = (data.ops ?? data) as AssemblyOpData[];
      if (this.scrubber && ops.length > 0) {
        this.scrubber.setOps(ops);
      }
    } catch (err) {
      console.warn('[dfm] assembly-sequence error:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // DFM analysis polling
  // ---------------------------------------------------------------------------

  private async _startDFMAnalysis(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${this.ctx.botName}/dfm/run`, { method: 'POST' });
      if (!resp.ok) {
        console.warn('[dfm] dfm/run failed:', resp.status);
        return;
      }
      const data = await resp.json();
      this.runId = data.run_id;
      if (this.panel) {
        this.panel.setProgress('Starting', 0, 1);
      }
      this._startPolling();
    } catch (err) {
      console.warn('[dfm] dfm/run error:', err);
    }
  }

  private _startPolling(): void {
    this._stopPolling();
    this.pollTimer = window.setInterval(() => this._pollStatus(), 500);
  }

  private _stopPolling(): void {
    if (this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private async _pollStatus(): Promise<void> {
    if (!this.runId) return;

    try {
      const resp = await fetch(`/api/bots/${this.ctx.botName}/dfm/${this.runId}/status`);
      if (!resp.ok) return;

      const data = await resp.json();
      const state: string = data.state ?? 'running';
      const checksComplete: number = data.checks_complete ?? 0;
      const checksTotal: number = data.checks_total ?? 1;
      const findings: DFMFindingData[] = data.findings ?? [];

      // Update progress
      if (this.panel) {
        this.panel.setProgress(state, checksComplete, checksTotal);
      }

      // Update findings (accumulate as they arrive)
      if (findings.length > 0) {
        this.findings = findings;
        if (this.panel) {
          this.panel.setFindings(findings);
          // Re-apply step filter if scrubber is active
          if (this.scrubber) {
            this.panel.filterByStep(this.scrubber.currentStep());
          }
        }
      }

      // Stop polling when complete
      if (state === 'complete' || state === 'error') {
        this._stopPolling();
        if (this.panel && findings.length > 0) {
          this.panel.setFindings(findings);
        }
      }
    } catch (err) {
      console.warn('[dfm] poll error:', err);
    }
  }

  // ---------------------------------------------------------------------------
  // Interaction handlers
  // ---------------------------------------------------------------------------

  private _onFindingClick(finding: DFMFindingData): void {
    // If finding has a body, isolate it and fly camera there
    const bodyId = this.bodyNameToId[finding.body];
    if (bodyId !== undefined) {
      // Ghost everything except the affected body
      this.ctx.botScene.ghostAllExcept([bodyId]);
      this.ctx.syncScene();

      // Fly camera to the body
      this.focus.focusOnBody(bodyId, 0.6);
    } else {
      // No specific body — show all, fly to whole bot
      this.ctx.botScene.unghost();
      this.ctx.syncScene();
      this.focus.focusOnAll(0.6);
    }
  }

  private _onStepChange(step: number): void {
    if (this.panel) {
      this.panel.filterByStep(step);
    }
  }
}
