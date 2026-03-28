/**
 * DFM Viewer — top-level Design for Manufacturing analysis viewer.
 *
 * Loads meshes from the assembly sequence API response. Each assembly op
 * carries mesh references (file, pos, quat) so the assembly sequence is the
 * single source of truth for both DFM checks and the viewer's incremental
 * build-up visualization.
 *
 * Entry point: initDFMViewer(botName, viewport, sidePanelEl)
 */

import * as THREE from 'three';
import type { AssemblyMeshRef, AssemblyOpData } from './assembly-scrubber.ts';
import { AssemblyScrubber } from './assembly-scrubber.ts';
import { type DFMFindingData, DFMPanel } from './dfm-panel.ts';
import type { ManifestMaterial, ViewerManifest } from './manifest-types.ts';
import { fetchSTL, makeMaterial, manifestQuatToThree } from './utils.ts';
import type { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface DFMHandle {
  pause(): void;
  resume(): void;
  dispose(): void;
}

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
    // Re-enable damping now that the animation is done
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

  // Camera direction: maintain current viewing angle
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
  const dist = maxDim * 2.5;

  anim.startPos = camera.position.clone();
  anim.endPos = new THREE.Vector3().addVectors(center, dir.multiplyScalar(dist));
  anim.startTarget = controls.target.clone();
  anim.endTarget = center.clone();
  anim.startTime = performance.now();
  anim.duration = duration;
  anim.active = true;

  // Disable damping for the duration of the animation so OrbitControls
  // doesn't fight our per-frame position interpolation. The viewport's
  // _tick() calls controls.update() with damping before our callback,
  // which would partially undo each animation step.
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
  // Named material from manifest materials dict (multi-material components)
  if (ref.material && materials[ref.material]) {
    return makeMaterial(materials[ref.material]);
  }
  // Fastener — metallic silver
  if (action === 'fasten') {
    return new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.2 });
  }
  // Wire — use color if provided
  if (action === 'route_wire' && ref.color) {
    const c = new THREE.Color(ref.color[0], ref.color[1], ref.color[2]);
    return new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5, emissive: c, emissiveIntensity: 0.15 });
  }
  // Color from the mesh ref (body shell or component color)
  if (ref.color) {
    return new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(ref.color[0], ref.color[1], ref.color[2]),
      roughness: 0.6,
      metalness: 0.0,
    });
  }
  // Default gray for components
  return new THREE.MeshPhysicalMaterial({ color: 0xd9d9d9, roughness: 0.6, metalness: 0.0 });
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initDFMViewer(
  botName: string,
  viewport: Viewport3D,
  sidePanelEl: HTMLElement,
): Promise<DFMHandle> {
  // Fetch manifest for materials dict only (mesh loading is driven by assembly ops)
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

  // Single group for all DFM meshes
  const dfmGroup = viewport.addGroup('dfm-meshes');

  // ---------------------------------------------------------------------------
  // DFM panel + scrubber
  // ---------------------------------------------------------------------------

  let panel: DFMPanel | null = null;
  let scrubber: AssemblyScrubber | null = null;
  let pollTimer: number | null = null;
  let runId: string | null = null;
  let findings: DFMFindingData[] = [];
  let assemblyOps: AssemblyOpData[] = [];

  // Mount DFM panel in the side panel
  sidePanelEl.innerHTML = '';
  panel = new DFMPanel(sidePanelEl, (finding) => onFindingClick(finding));

  // Mount scrubber in canvas container
  const canvasContainer = viewport.renderer.domElement.parentElement;
  if (canvasContainer) {
    scrubber = new AssemblyScrubber(canvasContainer, (step) => onStepChange(step));
  }

  // ---------------------------------------------------------------------------
  // Assembly sequence loading — meshes come from ops
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

            // Start hidden — onStepChange will reveal
            mesh.visible = false;

            dfmGroup.add(mesh);

            // Track by step
            if (!stepMeshes.has(op.step)) {
              stepMeshes.set(op.step, []);
            }
            stepMeshes.get(op.step)!.push(mesh);

            // Track by body for fly-to
            if (!bodyMeshes[op.body]) {
              bodyMeshes[op.body] = [];
            }
            bodyMeshes[op.body].push(mesh);
          })();
          loadPromises.push(promise);
        }
      }

      await Promise.all(loadPromises);

      // Show all meshes initially and frame camera
      showAllMeshes();
      const box = new THREE.Box3().expandByObject(dfmGroup);
      if (!box.isEmpty()) {
        viewport.frameOnBox(box);
      }

      // If scrubber exists, set to last step (show all)
      if (scrubber && ops.length > 0) {
        onStepChange(ops[ops.length - 1].step);
      }
    } catch (err) {
      console.warn('[dfm] assembly-sequence error:', err);
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

    // Reset all mesh opacities
    dfmGroup.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
        if (mat.opacity !== undefined) {
          mat.transparent = false;
          mat.opacity = 1.0;
        }
      }
    });

    if (meshesForBody && meshesForBody.length > 0) {
      // Ghost everything except the affected body's meshes
      const targetSet = new Set(meshesForBody);
      dfmGroup.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && !targetSet.has(child as THREE.Mesh)) {
          const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
          mat.transparent = true;
          mat.opacity = 0.06;
        }
      });

      // Fly camera to the body's bounding box
      const bodyBox = new THREE.Box3();
      for (const m of meshesForBody) {
        bodyBox.expandByObject(m);
      }
      if (!bodyBox.isEmpty()) {
        flyToBox(bodyBox, viewport.camera, viewport.controls as any, cameraAnim);
      }
    } else {
      // No specific body — fly to whole bot
      const allBox = new THREE.Box3().expandByObject(dfmGroup);
      if (!allBox.isEmpty()) {
        flyToBox(allBox, viewport.camera, viewport.controls as any, cameraAnim);
      }
    }
  }

  function onStepChange(step: number): void {
    if (panel) {
      panel.filterByStep(step);
    }

    if (assemblyOps.length === 0) return;

    // Show meshes for ops 0..step, hide the rest
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

      if (panel) {
        panel.setProgress(state, checksComplete, checksTotal);
      }

      // Findings live at a separate endpoint, not in the status response
      if (checksComplete > 0) {
        try {
          const findingsResp = await fetch(`/api/bots/${botName}/dfm/${runId}/findings`);
          if (findingsResp.ok) {
            const findingsData = await findingsResp.json();
            const newFindings: DFMFindingData[] = findingsData.findings ?? [];
            if (newFindings.length > 0) {
              findings = newFindings;
              if (panel) {
                panel.setFindings(findings);
                if (scrubber) {
                  panel.filterByStep(scrubber.currentStep());
                }
              }
            }
          }
        } catch (findingsErr) {
          console.warn('[dfm] findings fetch error:', findingsErr);
        }
      }

      if (state === 'complete' || state === 'error') {
        stopPolling();
      }
    } catch (err) {
      console.warn('[dfm] poll error:', err);
    }
  }

  async function startDFMAnalysis(): Promise<void> {
    try {
      const resp = await fetch(`/api/bots/${botName}/dfm/run`, { method: 'POST' });
      if (!resp.ok) return;
      const data = await resp.json();
      runId = data.run_id;
      if (panel) {
        panel.setProgress('Starting', 0, 1);
      }
      stopPolling();
      pollTimer = window.setInterval(() => pollStatus(), 500);
    } catch (err) {
      console.warn('[dfm] dfm/run error:', err);
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
      if (panel) {
        panel.dispose();
        panel = null;
      }
      if (scrubber) {
        scrubber.dispose();
        scrubber = null;
      }
      viewport.scene.remove(dfmGroup);
    },
  };
}
