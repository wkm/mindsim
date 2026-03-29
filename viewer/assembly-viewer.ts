/**
 * Assembly Viewer — data-first layout with nav/detail/viewport split.
 *
 * Layout:
 *   Top third:  sub-tab bar (Steps / DAG / DFM) + nav pane (scrollable)
 *   Bottom 2/3: left = detail panel, right = 3D viewport
 *
 * All three sub-tabs share the same detail panel and 3D viewport. Selecting
 * a step in any mode updates both the detail pane and mesh visibility.
 *
 * Entry point: initAssemblyViewer(botName)
 */

import * as THREE from 'three';
import { AssemblyDAG } from './assembly-dag.ts';
import type { AssemblyMeshRef, AssemblyOpData } from './assembly-scrubber.ts';
import { type DFMFindingData, DFMPanel } from './dfm-panel.ts';
import type { ManifestMaterial, ViewerManifest } from './manifest-types.ts';
import { fetchSTL, makeMaterial, manifestQuatToThree } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface AssemblyHandle {
  pause(): void;
  resume(): void;
  dispose(): void;
  /** The root DOM element — caller should mount this in the content area. */
  rootEl: HTMLElement;
}

// ---------------------------------------------------------------------------
// Sub-tab type
// ---------------------------------------------------------------------------

type AssemblySubTab = 'steps' | 'dag' | 'dfm';

// ---------------------------------------------------------------------------
// Mesh helpers
// ---------------------------------------------------------------------------

const EDGE_LINE_MAT = new THREE.LineBasicMaterial({
  color: 0x000000,
  transparent: true,
  opacity: 0.35,
  linewidth: 2,
});

function createPositionedMesh(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  pos: number[],
  quat: number[],
): THREE.Mesh {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.position.set(pos[0], pos[1], pos[2]);
  mesh.quaternion.copy(manifestQuatToThree(quat));

  const edges = new THREE.EdgesGeometry(geometry, 28);
  const lines = new THREE.LineSegments(edges, EDGE_LINE_MAT);
  lines.raycast = () => {};
  mesh.add(lines);

  return mesh;
}

// ---------------------------------------------------------------------------
// Camera fly-to animation (standalone, no MuJoCo dependency)
// ---------------------------------------------------------------------------

interface CameraAnimControls {
  target: THREE.Vector3;
  enableDamping?: boolean;
  update(): void;
}

interface CameraAnim {
  active: boolean;
  startTime: number;
  duration: number;
  startPos: THREE.Vector3;
  endPos: THREE.Vector3;
  startTarget: THREE.Vector3;
  endTarget: THREE.Vector3;
  controls: CameraAnimControls | null;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
}

function updateCameraAnim(anim: CameraAnim, camera: THREE.Camera): void {
  if (!anim.active || !anim.controls) return;
  const elapsed = (performance.now() - anim.startTime) / 1000;
  const t = Math.min(elapsed / anim.duration, 1);
  const e = easeInOut(t);

  camera.position.lerpVectors(anim.startPos, anim.endPos, e);
  anim.controls.target.lerpVectors(anim.startTarget, anim.endTarget, e);
  anim.controls.update();

  if (t >= 1) {
    anim.active = false;
    anim.controls.enableDamping = true;
    anim.controls = null;
  }
}

function flyToBox(
  box: THREE.Box3,
  camera: THREE.Camera,
  controls: CameraAnimControls,
  anim: CameraAnim,
  duration = 0.6,
): void {
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 0.1);

  const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const dist = maxDim * 2.5;

  anim.startPos = camera.position.clone();
  anim.endPos = new THREE.Vector3().addVectors(center, dir.multiplyScalar(dist));
  anim.startTarget = controls.target.clone();
  anim.endTarget = center.clone();
  anim.startTime = performance.now();
  anim.duration = duration;
  anim.active = true;

  anim.controls = controls;
  controls.enableDamping = false;
}

// ---------------------------------------------------------------------------
// Material helpers
// ---------------------------------------------------------------------------

function materialForMeshRef(
  ref: AssemblyMeshRef,
  action: string,
  materials: Record<string, ManifestMaterial>,
): THREE.Material {
  if (ref.material && materials[ref.material]) {
    return makeMaterial(materials[ref.material]);
  }
  if (action === 'fasten') {
    return new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.2 });
  }
  if (action === 'route_wire' && ref.color) {
    const c = new THREE.Color(ref.color[0], ref.color[1], ref.color[2]);
    return new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5, emissive: c, emissiveIntensity: 0.15 });
  }
  if (ref.color) {
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(ref.color[0], ref.color[1], ref.color[2]),
      roughness: 0.6,
      metalness: 0.0,
    });
  }
  return new THREE.MeshPhysicalMaterial({ color: 0xd9d9d9, roughness: 0.6, metalness: 0.0 });
}

// ---------------------------------------------------------------------------
// Repr syntax highlighting (duplicated from assembly-scrubber to avoid export)
// ---------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightRepr(repr: string): string {
  let s = escapeHtml(repr);

  const tokens: { key: string; html: string }[] = [];
  let tokenId = 0;
  const placeholder = (cls: string, text: string) => {
    const key = `\x00T${tokenId++}\x00`;
    tokens.push({ key, html: `<span style="${cls}">${text}</span>` });
    return key;
  };

  s = s.replace(/'[^']*'/g, (m) => placeholder('color:#D4A72C', m));
  s = s.replace(/"[^"]*"/g, (m) => placeholder('color:#D4A72C', m));
  s = s.replace(/\b(AssemblyOp|AssemblyAction|ComponentRef|FastenerRef|WireRef|JointId|BodyId|ToolKind)\b/g, (m) =>
    placeholder('color:#2B95D6;font-weight:600', m),
  );
  s = s.replace(/\.([A-Z][A-Z0-9_]+)\b/g, (_m, val) => `.${placeholder('color:#0F9960;font-weight:600', val)}`);
  s = s.replace(/\b(-?\d+\.?\d*)\b/g, (m) => placeholder('color:#C87619', m));
  s = s.replace(/\b(None|True|False)\b/g, (m) => placeholder('color:#9179F2', m));

  for (const { key, html } of tokens) {
    s = s.replace(key, html);
  }

  return s;
}

// ---------------------------------------------------------------------------
// Styles (injected once)
// ---------------------------------------------------------------------------

let stylesInjected = false;

function injectAssemblyStyles(): void {
  if (stylesInjected) return;
  stylesInjected = true;

  const style = document.createElement('style');
  style.textContent = `
    /* Assembly root layout */
    .assembly-root {
      display: flex;
      flex-direction: column;
      height: 100%;
      background: var(--background);
    }

    /* Nav pane (top third) */
    .assembly-nav {
      height: 33%;
      min-height: 120px;
      overflow-y: auto;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      padding-top: 4px;
    }

    /* Bottom 2/3 */
    .assembly-bottom {
      flex: 1;
      display: flex;
      min-height: 0;
    }

    /* Detail pane (bottom left) */
    .assembly-detail {
      width: 50%;
      overflow-y: auto;
      padding: 12px;
      border-right: 1px solid var(--border);
    }

    /* Viewport pane (bottom right) */
    .assembly-viewport {
      width: 50%;
      position: relative;
      min-height: 0;
    }

    /* Steps list */
    .asm-step-row {
      font-family: "Input Mono Narrow", "SF Mono", "Menlo", monospace;
      font-size: 11px;
      line-height: 1.6;
      padding: 2px 12px;
      cursor: pointer;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: var(--muted-fg);
      border-left: 3px solid transparent;
      transition: background 0.08s;
    }
    .asm-step-row:hover {
      background: var(--secondary);
    }
    .asm-step-row.selected {
      background: var(--secondary);
      color: var(--foreground);
      border-left-color: var(--primary);
      font-weight: 600;
    }

    /* DFM nav list */
    .asm-dfm-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font);
      font-size: 11px;
      line-height: 1.6;
      padding: 3px 12px;
      cursor: pointer;
      border-left: 3px solid transparent;
      transition: background 0.08s;
    }
    .asm-dfm-row:hover {
      background: var(--secondary);
    }
    .asm-dfm-row.selected {
      background: var(--secondary);
      border-left-color: var(--primary);
    }
    .asm-dfm-row.dimmed {
      opacity: 0.5;
    }
    .asm-dfm-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      border-radius: 9px;
      font-size: 10px;
      font-weight: 700;
      padding: 0 5px;
      flex-shrink: 0;
    }
    .asm-dfm-badge.error {
      background: var(--destructive, #e53e3e);
      color: #fff;
    }
    .asm-dfm-badge.warning {
      background: var(--gold3, #d69e2e);
      color: #fff;
    }
    .asm-dfm-badge.pass {
      background: var(--success, #38a169);
      color: #fff;
    }

    /* Detail pane repr block */
    .asm-detail-repr {
      margin: 0;
      padding: 10px 12px;
      background: #1C2127;
      border-radius: 6px;
      font-family: "Input Mono Narrow", "SF Mono", "Menlo", monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #ABB3BF;
      overflow-x: auto;
      max-height: 50%;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }

    /* Detail pane findings section */
    .asm-detail-findings {
      margin-top: 12px;
    }
    .asm-detail-findings h4 {
      font-family: var(--font);
      font-size: 11px;
      font-weight: 600;
      color: var(--muted-fg);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin: 0 0 8px 0;
    }
    .asm-finding-card {
      padding: 8px 10px;
      margin-bottom: 6px;
      border-radius: 4px;
      border-left: 3px solid var(--border);
      background: var(--card);
      font-family: var(--font);
      font-size: 11px;
      cursor: pointer;
    }
    .asm-finding-card:hover {
      background: var(--secondary);
    }
    .asm-finding-card.severity-error {
      border-left-color: var(--destructive, #e53e3e);
    }
    .asm-finding-card.severity-warning {
      border-left-color: var(--gold3, #d69e2e);
    }
    .asm-finding-card.severity-info {
      border-left-color: var(--primary);
    }
    .asm-finding-title {
      font-weight: 600;
      color: var(--foreground);
    }
    .asm-finding-desc {
      color: var(--muted-fg);
      margin-top: 2px;
    }
  `;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Single-line op summary for the Steps list
// ---------------------------------------------------------------------------

function opSummary(op: AssemblyOpData): string {
  return op.repr_oneline ?? op.repr ?? `step ${op.step}: ${op.action} ${op.body}`;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initAssemblyViewer(botName: string): Promise<AssemblyHandle> {
  injectAssemblyStyles();

  // =========================================================================
  // Build DOM layout
  // =========================================================================

  const root = document.createElement('div');
  root.className = 'assembly-root';

  // -- Nav pane (top third) --
  const navPane = document.createElement('div');
  navPane.className = 'assembly-nav';
  root.appendChild(navPane);

  // -- Bottom 2/3 --
  const bottomPane = document.createElement('div');
  bottomPane.className = 'assembly-bottom';
  root.appendChild(bottomPane);

  const detailPane = document.createElement('div');
  detailPane.className = 'assembly-detail';
  bottomPane.appendChild(detailPane);

  const viewportPane = document.createElement('div');
  viewportPane.className = 'assembly-viewport';
  bottomPane.appendChild(viewportPane);

  // =========================================================================
  // Create Viewport3D inside viewportPane
  // =========================================================================

  const viewport = new Viewport3D(viewportPane, {
    cameraType: 'perspective',
    grid: true,
  });

  // =========================================================================
  // Fetch manifest
  // =========================================================================

  const manifestResp = await fetch(`/api/bots/${botName}/viewer_manifest`);
  const manifest: ViewerManifest | null = manifestResp.ok ? await manifestResp.json() : null;
  const materials: Record<string, ManifestMaterial> = manifest?.materials ?? {};

  // Camera animation state
  const cameraAnim: CameraAnim = {
    active: false,
    startTime: 0,
    duration: 0,
    startPos: new THREE.Vector3(),
    endPos: new THREE.Vector3(),
    startTarget: new THREE.Vector3(),
    endTarget: new THREE.Vector3(),
    controls: null,
  };

  // Meshes tracked per assembly step for visibility toggling
  const stepMeshes: Map<number, THREE.Mesh[]> = new Map();

  // Body name -> meshes lookup for fly-to on finding click
  const bodyMeshes: Record<string, THREE.Mesh[]> = {};

  // Single group for all assembly meshes
  const asmGroup = viewport.addGroup('assembly-meshes');

  // =========================================================================
  // Shared state
  // =========================================================================

  let dfmPanel: DFMPanel | null = null;
  let dag: AssemblyDAG | null = null;
  let pollTimer: number | null = null;
  let runId: string | null = null;
  let findings: DFMFindingData[] = [];
  let assemblyOps: AssemblyOpData[] = [];
  let activeSubTab: AssemblySubTab = 'steps';
  let selectedStep = -1;

  // =========================================================================
  // Nav pane containers (one per sub-tab, swapped in/out)
  // =========================================================================

  // Steps list container
  const stepsListEl = document.createElement('div');
  stepsListEl.style.cssText = 'padding: 4px 0;';

  // DAG container
  const dagContainer = document.createElement('div');
  dagContainer.style.cssText = 'width: 100%; height: 100%; overflow: auto;';
  dag = new AssemblyDAG(dagContainer, (step) => selectStep(step));

  // DFM nav list container
  const dfmNavEl = document.createElement('div');
  dfmNavEl.style.cssText = 'padding: 4px 0;';

  // DFM findings panel (used internally for data, not for direct DOM display)
  const dfmInternalContainer = document.createElement('div');
  dfmPanel = new DFMPanel(dfmInternalContainer, (finding) => onFindingClick(finding));

  // =========================================================================
  // Steps list rendering
  // =========================================================================

  function renderStepsList(): void {
    stepsListEl.innerHTML = '';
    for (const op of assemblyOps) {
      const row = document.createElement('div');
      row.className = `asm-step-row${op.step === selectedStep ? ' selected' : ''}`;
      row.innerHTML = highlightRepr(opSummary(op));
      row.addEventListener('click', () => selectStep(op.step));
      stepsListEl.appendChild(row);
    }
  }

  // Keyboard navigation: arrow up/down to move through steps
  root.tabIndex = 0;
  root.style.outline = 'none';
  root.addEventListener('keydown', (e: KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = Math.min(selectedStep + 1, assemblyOps.length - 1);
      selectStep(next);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = Math.max(selectedStep - 1, 0);
      selectStep(prev);
    }
  });

  function updateStepsListSelection(): void {
    const rows = stepsListEl.querySelectorAll('.asm-step-row');
    for (let i = 0; i < rows.length; i++) {
      rows[i].classList.toggle('selected', i < assemblyOps.length && assemblyOps[i].step === selectedStep);
    }
    // Auto-scroll to selected
    const sel = stepsListEl.querySelector('.asm-step-row.selected') as HTMLElement | null;
    if (sel) {
      sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // =========================================================================
  // DFM nav list rendering
  // =========================================================================

  function findingsForStep(step: number): DFMFindingData[] {
    return findings.filter((f) => f.assembly_step === step);
  }

  function renderDfmNavList(): void {
    dfmNavEl.innerHTML = '';
    for (const op of assemblyOps) {
      const stepFindings = findingsForStep(op.step);
      const errorCount = stepFindings.filter((f) => f.severity === 'error').length;
      const warnCount = stepFindings.filter((f) => f.severity === 'warning').length;
      const hasFindings = stepFindings.length > 0;

      const row = document.createElement('div');
      row.className = `asm-dfm-row${op.step === selectedStep ? ' selected' : ''}${!hasFindings ? ' dimmed' : ''}`;

      // Badge
      const badge = document.createElement('span');
      if (errorCount > 0) {
        badge.className = 'asm-dfm-badge error';
        badge.textContent = String(errorCount);
      } else if (warnCount > 0) {
        badge.className = 'asm-dfm-badge warning';
        badge.textContent = String(warnCount);
      } else {
        badge.className = 'asm-dfm-badge pass';
        badge.textContent = '\u2713';
      }
      row.appendChild(badge);

      // Label
      const label = document.createElement('span');
      label.textContent = opSummary(op);
      label.style.cssText = 'overflow: hidden; text-overflow: ellipsis; white-space: nowrap;';
      row.appendChild(label);

      row.addEventListener('click', () => selectStep(op.step));
      dfmNavEl.appendChild(row);
    }
  }

  function updateDfmNavSelection(): void {
    const rows = dfmNavEl.querySelectorAll('.asm-dfm-row');
    for (let i = 0; i < rows.length; i++) {
      const isSelected = i < assemblyOps.length && assemblyOps[i].step === selectedStep;
      rows[i].classList.toggle('selected', isSelected);
    }
    const sel = dfmNavEl.querySelector('.asm-dfm-row.selected') as HTMLElement | null;
    if (sel) {
      sel.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // =========================================================================
  // Detail pane rendering
  // =========================================================================

  function renderDetail(): void {
    detailPane.innerHTML = '';
    const op = assemblyOps.find((o) => o.step === selectedStep);
    if (!op) {
      const empty = document.createElement('div');
      empty.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
      empty.textContent = 'Select an assembly step to view details.';
      detailPane.appendChild(empty);
      return;
    }

    // Pretty repr
    const repr = document.createElement('pre');
    repr.className = 'asm-detail-repr';
    if (op.repr) {
      repr.innerHTML = highlightRepr(op.repr);
    } else {
      const toolStr = op.tool ? ` [${op.tool}]` : '';
      repr.textContent = `${op.action}: ${op.body} \u2014 ${op.description}${toolStr}`;
    }
    detailPane.appendChild(repr);

    // In DFM mode (or always when findings exist for this step): show findings
    const stepFindings = findingsForStep(op.step);
    if (stepFindings.length > 0) {
      const section = document.createElement('div');
      section.className = 'asm-detail-findings';

      const heading = document.createElement('h4');
      heading.textContent = `Findings (${stepFindings.length})`;
      section.appendChild(heading);

      for (const f of stepFindings) {
        const card = document.createElement('div');
        card.className = `asm-finding-card severity-${f.severity}`;
        card.addEventListener('click', () => onFindingClick(f));

        const title = document.createElement('div');
        title.className = 'asm-finding-title';
        title.textContent = `${f.severity.toUpperCase()}: ${f.title}`;
        card.appendChild(title);

        const desc = document.createElement('div');
        desc.className = 'asm-finding-desc';
        desc.textContent = f.description;
        card.appendChild(desc);

        section.appendChild(card);
      }

      detailPane.appendChild(section);
    }
  }

  // =========================================================================
  // Step selection (shared across all sub-tabs)
  // =========================================================================

  function selectStep(step: number): void {
    selectedStep = step;

    // Update mesh visibility
    for (const [opStep, meshes] of stepMeshes.entries()) {
      const visible = opStep <= step;
      for (const m of meshes) m.visible = visible;
    }

    // Update detail pane
    renderDetail();

    // Update nav list selections
    updateStepsListSelection();
    updateDfmNavSelection();

    // Update DAG highlight
    if (dag) dag.highlightStep(step);

    // Update DFM panel internal filter
    if (dfmPanel) dfmPanel.filterByStep(step);
  }

  // =========================================================================
  // Sub-tab switching
  // =========================================================================

  function switchSubTab(tab: AssemblySubTab): void {
    if (tab === activeSubTab) return;
    activeSubTab = tab;

    navPane.innerHTML = '';

    if (tab === 'steps') {
      navPane.appendChild(stepsListEl);
      updateStepsListSelection();
    } else if (tab === 'dag') {
      navPane.appendChild(dagContainer);
    } else if (tab === 'dfm') {
      navPane.appendChild(dfmNavEl);
      updateDfmNavSelection();
    }
  }

  // Initialize Steps as the active sub-tab
  navPane.appendChild(stepsListEl);

  // =========================================================================
  // Assembly sequence loading
  // =========================================================================

  async function loadAssemblySequence(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${botName}/assembly-sequence`);
      if (!resp.ok) return;
      const data = await resp.json();
      const ops: AssemblyOpData[] = (data.ops ?? data) as AssemblyOpData[];
      assemblyOps = ops;

      // Update DAG with ops
      if (dag && ops.length > 0) {
        dag.setOps(ops);
      }

      // Load all meshes referenced by assembly ops
      const loadPromises: Promise<void>[] = [];

      for (const op of ops) {
        const meshRefs = op.meshes ?? [];
        if (meshRefs.length === 0) continue;

        for (const ref of meshRefs) {
          const promise = (async () => {
            const geometry = await fetchSTL(botName, ref.file);
            if (!geometry) return;

            const mat = materialForMeshRef(ref, op.action, materials);
            const mesh = createPositionedMesh(geometry, mat, ref.pos, ref.quat);

            mesh.visible = false;
            asmGroup.add(mesh);

            if (!stepMeshes.has(op.step)) {
              stepMeshes.set(op.step, []);
            }
            stepMeshes.get(op.step)!.push(mesh);

            if (!bodyMeshes[op.body]) {
              bodyMeshes[op.body] = [];
            }
            bodyMeshes[op.body].push(mesh);
          })();
          loadPromises.push(promise);
        }
      }

      await Promise.all(loadPromises);

      // Show all meshes and frame
      for (const meshes of stepMeshes.values()) {
        for (const m of meshes) m.visible = true;
      }
      const box = new THREE.Box3().expandByObject(asmGroup);
      if (!box.isEmpty()) {
        viewport.frameOnBox(box);
      }

      // Render nav lists
      renderStepsList();
      renderDfmNavList();

      // Select last step
      if (ops.length > 0) {
        selectStep(ops[ops.length - 1].step);
      }
    } catch (err) {
      console.warn('[assembly] assembly-sequence error:', err);
    }
  }

  // =========================================================================
  // Finding click: ghost other bodies, fly camera to affected body
  // =========================================================================

  function onFindingClick(finding: DFMFindingData): void {
    // First select the step so mesh visibility is correct
    selectStep(finding.assembly_step);

    const meshesForBody = bodyMeshes[finding.body];

    asmGroup.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
        if (mat.opacity !== undefined) {
          mat.transparent = false;
          mat.opacity = 1.0;
        }
      }
    });

    if (meshesForBody && meshesForBody.length > 0) {
      const targetSet = new Set(meshesForBody);
      asmGroup.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && !targetSet.has(child as THREE.Mesh)) {
          const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
          mat.transparent = true;
          mat.opacity = 0.06;
        }
      });

      const bodyBox = new THREE.Box3();
      for (const m of meshesForBody) {
        bodyBox.expandByObject(m);
      }
      if (!bodyBox.isEmpty()) {
        flyToBox(bodyBox, viewport.camera, viewport.controls as CameraAnimControls, cameraAnim);
      }
    } else {
      const allBox = new THREE.Box3().expandByObject(asmGroup);
      if (!allBox.isEmpty()) {
        flyToBox(allBox, viewport.camera, viewport.controls as CameraAnimControls, cameraAnim);
      }
    }
  }

  // =========================================================================
  // DFM analysis polling
  // =========================================================================

  function stopPolling(): void {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function pollStatus(): Promise<void> {
    if (!runId) return;
    try {
      const resp = await fetch(`/api/bots/${botName}/dfm/${runId}/status`);
      if (!resp.ok) return;

      const data = await resp.json();
      const state: string = data.state ?? 'running';
      const checksComplete: number = data.checks_complete ?? 0;
      const checksTotal: number = data.checks_total ?? 1;

      if (dfmPanel) {
        dfmPanel.setProgress(state, checksComplete, checksTotal);
      }

      if (checksComplete > 0) {
        try {
          const findingsResp = await fetch(`/api/bots/${botName}/dfm/${runId}/findings`);
          if (findingsResp.ok) {
            const findingsData = await findingsResp.json();
            const newFindings: DFMFindingData[] = findingsData.findings ?? [];
            if (newFindings.length > 0) {
              findings = newFindings;
              if (dfmPanel) {
                dfmPanel.setFindings(findings);
                dfmPanel.filterByStep(selectedStep);
              }
              // Re-render DFM nav list with updated findings
              renderDfmNavList();
              // Re-render detail if we have findings for the current step
              renderDetail();
            }
          }
        } catch (findingsErr) {
          console.warn('[assembly] findings fetch error:', findingsErr);
        }
      }

      if (state === 'complete' || state === 'error') {
        stopPolling();
      }
    } catch (err) {
      console.warn('[assembly] poll error:', err);
    }
  }

  async function startDFMAnalysis(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${botName}/dfm/run`, { method: 'POST' });
      if (!resp.ok) return;
      const data = await resp.json();
      runId = data.run_id;
      if (dfmPanel) {
        dfmPanel.setProgress('Starting', 0, 1);
      }
      stopPolling();
      pollTimer = window.setInterval(() => pollStatus(), 500);
    } catch (err) {
      console.warn('[assembly] dfm/run error:', err);
    }
  }

  // Kick off data loading
  loadAssemblySequence();
  startDFMAnalysis();

  // =========================================================================
  // Animation loop (camera fly-to)
  // =========================================================================

  let paused = false;

  viewport.animate(() => {
    if (paused) return;
    updateCameraAnim(cameraAnim, viewport.camera);
  });

  // Resize viewport when container becomes visible (deferred mount)
  const resizeObserver = new ResizeObserver(() => {
    viewport.resize();
  });
  resizeObserver.observe(viewportPane);

  // =========================================================================
  // Handle
  // =========================================================================

  return {
    rootEl: root,
    pause() {
      paused = true;
    },
    resume() {
      paused = false;
      viewport.resize();
    },
    dispose() {
      stopPolling();
      resizeObserver.disconnect();
      if (dfmPanel) {
        dfmPanel.dispose();
        dfmPanel = null;
      }
      if (dag) {
        dag.dispose();
        dag = null;
      }
      viewport.scene.remove(asmGroup);
      viewport.dispose();
      root.remove();
    },
  };
}
