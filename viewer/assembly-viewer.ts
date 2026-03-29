/**
 * Assembly Viewer — top-level Assembly tab with Steps / DAG / DFM sub-views.
 *
 * All three sub-views share the same 3D viewport and mesh state. Only the
 * side panel / bottom bar content changes between sub-tabs.
 *
 * - Steps: scrubber + incremental mesh build-up + op repr display
 * - DAG: prerequisite graph visualization (SVG)
 * - DFM: findings table + progress indicator
 *
 * Entry point: initAssemblyViewer(botName, viewport, sidePanelEl)
 */

import * as THREE from 'three';
import { AssemblyDAG } from './assembly-dag.ts';
import type { AssemblyMeshRef, AssemblyOpData } from './assembly-scrubber.ts';
import { AssemblyScrubber } from './assembly-scrubber.ts';
import { type DFMFindingData, DFMPanel } from './dfm-panel.ts';
import type { ManifestMaterial, ViewerManifest } from './manifest-types.ts';
import { fetchSTL, makeMaterial, manifestQuatToThree } from './utils.ts';
import type { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface AssemblyHandle {
  pause(): void;
  resume(): void;
  dispose(): void;
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
// Styles (injected once)
// ---------------------------------------------------------------------------

let stylesInjected = false;

function injectSubTabStyles(): void {
  if (stylesInjected) return;
  stylesInjected = true;

  const style = document.createElement('style');
  style.textContent = `
    .assembly-subtabs {
      display: flex;
      gap: 2px;
      padding: 4px 8px;
      background: var(--card);
      border-bottom: 1px solid var(--border);
    }
    .assembly-subtab {
      font-family: var(--font);
      font-size: 11px;
      font-weight: 500;
      padding: 4px 12px;
      border: none;
      border-radius: var(--radius-sm);
      background: transparent;
      color: var(--muted-fg);
      cursor: pointer;
      transition: background 0.1s, color 0.1s;
    }
    .assembly-subtab:hover {
      background: var(--secondary);
      color: var(--foreground);
    }
    .assembly-subtab.active {
      background: var(--primary);
      color: var(--primary-fg, #fff);
    }
  `;
  document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initAssemblyViewer(
  botName: string,
  viewport: Viewport3D,
  sidePanelEl: HTMLElement,
): Promise<AssemblyHandle> {
  injectSubTabStyles();

  // Fetch manifest for materials dict only
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

  // ---------------------------------------------------------------------------
  // Shared state
  // ---------------------------------------------------------------------------

  let dfmPanel: DFMPanel | null = null;
  let scrubber: AssemblyScrubber | null = null;
  let dag: AssemblyDAG | null = null;
  let pollTimer: number | null = null;
  let runId: string | null = null;
  let findings: DFMFindingData[] = [];
  let assemblyOps: AssemblyOpData[] = [];
  let activeSubTab: AssemblySubTab = 'steps';

  // ---------------------------------------------------------------------------
  // Sub-tab bar (rendered inside the side panel)
  // ---------------------------------------------------------------------------

  const subTabBar = document.createElement('div');
  subTabBar.className = 'assembly-subtabs';

  const subTabButtons: Record<AssemblySubTab, HTMLButtonElement> = {} as Record<AssemblySubTab, HTMLButtonElement>;
  for (const tab of ['steps', 'dag', 'dfm'] as const) {
    const btn = document.createElement('button');
    btn.className = `assembly-subtab${tab === 'steps' ? ' active' : ''}`;
    btn.textContent = tab === 'dfm' ? 'DFM' : tab === 'dag' ? 'DAG' : 'Steps';
    btn.addEventListener('click', () => switchSubTab(tab));
    subTabBar.appendChild(btn);
    subTabButtons[tab] = btn;
  }

  // Sub-tab content container (below the tab bar, inside side panel)
  const subTabContent = document.createElement('div');
  subTabContent.style.cssText = 'flex: 1; overflow-y: auto;';

  // ---------------------------------------------------------------------------
  // Build side panel structure
  // ---------------------------------------------------------------------------

  sidePanelEl.innerHTML = '';
  sidePanelEl.style.display = 'flex';
  sidePanelEl.style.flexDirection = 'column';
  sidePanelEl.appendChild(subTabBar);
  sidePanelEl.appendChild(subTabContent);

  // ---------------------------------------------------------------------------
  // Steps sub-view: scrubber
  // ---------------------------------------------------------------------------

  const canvasContainer = viewport.renderer.domElement.parentElement;
  if (canvasContainer) {
    scrubber = new AssemblyScrubber(canvasContainer, (step) => onStepChange(step));
  }

  // ---------------------------------------------------------------------------
  // DFM sub-view: panel
  // ---------------------------------------------------------------------------

  const dfmContainer = document.createElement('div');
  dfmPanel = new DFMPanel(dfmContainer, (finding) => onFindingClick(finding));

  // ---------------------------------------------------------------------------
  // DAG sub-view
  // ---------------------------------------------------------------------------

  const dagContainer = document.createElement('div');
  dagContainer.style.cssText = 'width: 100%; height: 100%; overflow: auto;';
  dag = new AssemblyDAG(dagContainer, (step) => {
    if (scrubber) scrubber.setStep(step);
    onStepChange(step);
  });

  // ---------------------------------------------------------------------------
  // Sub-tab switching
  // ---------------------------------------------------------------------------

  function switchSubTab(tab: AssemblySubTab): void {
    if (tab === activeSubTab) return;
    activeSubTab = tab;

    for (const [key, btn] of Object.entries(subTabButtons)) {
      btn.classList.toggle('active', key === tab);
    }

    // Clear sub-tab content
    subTabContent.innerHTML = '';

    // Show/hide scrubber based on sub-tab
    if (scrubber) {
      // Steps and DFM show the scrubber; DAG hides it
      const scrubberEl = canvasContainer?.querySelector('.assembly-scrubber') as HTMLElement | null;
      if (scrubberEl) {
        scrubberEl.style.display = tab === 'dag' ? 'none' : '';
      }
    }

    if (tab === 'steps') {
      // Steps: side panel shows a placeholder message (repr is in scrubber overlay)
      const stepsInfo = document.createElement('div');
      stepsInfo.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
      stepsInfo.textContent = 'Use the scrubber below the viewport to step through the assembly sequence.';
      subTabContent.appendChild(stepsInfo);
    } else if (tab === 'dag') {
      subTabContent.appendChild(dagContainer);
    } else if (tab === 'dfm') {
      subTabContent.appendChild(dfmContainer);
    }
  }

  // Initialize with Steps sub-tab content
  switchSubTab('steps');
  // Force steps active since switchSubTab early-returns if same tab
  activeSubTab = 'steps';
  const stepsInfo = document.createElement('div');
  stepsInfo.style.cssText = 'padding: 16px; color: var(--muted-fg); font-size: 12px;';
  stepsInfo.textContent = 'Use the scrubber below the viewport to step through the assembly sequence.';
  subTabContent.appendChild(stepsInfo);

  // ---------------------------------------------------------------------------
  // Assembly sequence loading
  // ---------------------------------------------------------------------------

  async function loadAssemblySequence(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${botName}/assembly-sequence`);
      if (!resp.ok) return;
      const data = await resp.json();
      const ops: AssemblyOpData[] = (data.ops ?? data) as AssemblyOpData[];
      assemblyOps = ops;

      if (scrubber && ops.length > 0) {
        scrubber.setOps(ops);
      }

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

      showAllMeshes();
      const box = new THREE.Box3().expandByObject(asmGroup);
      if (!box.isEmpty()) {
        viewport.frameOnBox(box);
      }

      if (scrubber && ops.length > 0) {
        onStepChange(ops[ops.length - 1].step);
      }
    } catch (err) {
      console.warn('[assembly] assembly-sequence error:', err);
    }
  }

  function showAllMeshes(): void {
    for (const meshes of stepMeshes.values()) {
      for (const m of meshes) m.visible = true;
    }
  }

  // ---------------------------------------------------------------------------
  // Finding click: ghost other bodies, fly camera to affected body
  // ---------------------------------------------------------------------------

  function onFindingClick(finding: DFMFindingData): void {
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

  function onStepChange(step: number): void {
    if (dfmPanel) {
      dfmPanel.filterByStep(step);
    }
    if (dag) {
      dag.highlightStep(step);
    }

    if (assemblyOps.length === 0) return;

    for (const [opStep, meshes] of stepMeshes.entries()) {
      const visible = opStep <= step;
      for (const m of meshes) m.visible = visible;
    }
  }

  // ---------------------------------------------------------------------------
  // DFM analysis polling
  // ---------------------------------------------------------------------------

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
                if (scrubber) {
                  dfmPanel.filterByStep(scrubber.currentStep());
                }
              }
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

  // ---------------------------------------------------------------------------
  // Animation loop (camera fly-to)
  // ---------------------------------------------------------------------------

  let paused = false;

  viewport.animate(() => {
    if (paused) return;
    updateCameraAnim(cameraAnim, viewport.camera);
  });

  // ---------------------------------------------------------------------------
  // Handle
  // ---------------------------------------------------------------------------

  return {
    pause() {
      paused = true;
    },
    resume() {
      paused = false;
      viewport.resize();
    },
    dispose() {
      stopPolling();
      if (dfmPanel) {
        dfmPanel.dispose();
        dfmPanel = null;
      }
      if (scrubber) {
        scrubber.dispose();
        scrubber = null;
      }
      if (dag) {
        dag.dispose();
        dag = null;
      }
      viewport.scene.remove(asmGroup);
    },
  };
}
