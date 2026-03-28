/**
 * DFM Viewer — top-level Design for Manufacturing analysis viewer.
 *
 * Loads the same static meshes as the Design viewer, overlays the DFM
 * findings panel (side panel) and assembly step scrubber (bottom bar).
 * Fetches assembly sequence and DFM analysis results from the API.
 *
 * Entry point: initDFMViewer(botName, viewport, sidePanelEl)
 */

import * as THREE from 'three';
import { type AssemblyOpData, AssemblyScrubber } from './assembly-scrubber.ts';
import { type DFMFindingData, DFMPanel } from './dfm-panel.ts';
import type { ViewerManifest } from './manifest-types.ts';
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
// Mesh helpers (same as design-viewer.ts)
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
// Main entry point
// ---------------------------------------------------------------------------

export async function initDFMViewer(
  botName: string,
  viewport: Viewport3D,
  sidePanelEl: HTMLElement,
): Promise<DFMHandle> {
  // Fetch manifest (same as design-viewer)
  const resp = await fetch(`/api/bots/${botName}/viewer_manifest`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch viewer manifest: ${resp.status}`);
  }
  const manifest: ViewerManifest = await resp.json();
  const materials = manifest.materials ?? {};

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

  // Build body name -> meshes lookup for fly-to on finding click
  const bodyMeshes: Record<string, THREE.Mesh[]> = {};

  // Mount meshes keyed by "body:label" (matches ComponentRef in assembly ops)
  const mountMeshes: Record<string, THREE.Mesh[]> = {};

  // Part meshes keyed by "fastener:body:index" or "wire:label"
  const partMeshes: Record<string, THREE.Mesh[]> = {};

  // Create viewport groups per body
  const bodyGroups: Record<string, THREE.Group> = {};
  for (const body of manifest.bodies) {
    bodyGroups[body.name] = viewport.addGroup(`dfm-body:${body.name}`);
    bodyMeshes[body.name] = [];
  }

  function getBodyGroup(bodyName: string): THREE.Group {
    return bodyGroups[bodyName] ?? Object.values(bodyGroups)[0];
  }

  // Load body meshes
  const bodyMeshPromises = manifest.bodies.map(async (body) => {
    const geometry = await fetchSTL(botName, body.mesh);
    if (!geometry) return;

    const defaultColor = body.role === 'component' ? 0x808080 : 0xd9d9d9;
    const bodyColor = body.color
      ? new THREE.Color(body.color[0], body.color[1], body.color[2])
      : new THREE.Color(defaultColor);
    const mat = new THREE.MeshPhysicalMaterial({
      color: bodyColor,
      roughness: body.role === 'component' ? 0.5 : 0.6,
      metalness: 0.0,
    });

    const mesh = createPositionedMesh(geometry, mat, body.pos, body.quat);
    getBodyGroup(body.name).add(mesh);
    if (!bodyMeshes[body.name]) bodyMeshes[body.name] = [];
    bodyMeshes[body.name].push(mesh);
  });

  // Load mount meshes
  const mountsList = manifest.mounts ?? [];
  const mountMeshPromises = mountsList.map(async (mount) => {
    if (!mount.pos || !mount.quat) return;
    const group = getBodyGroup(mount.body);
    const mountKey = `${mount.body}:${mount.label}`;

    function trackMountMesh(m: THREE.Mesh): void {
      if (!mountMeshes[mountKey]) mountMeshes[mountKey] = [];
      mountMeshes[mountKey].push(m);
      if (!bodyMeshes[mount.body]) bodyMeshes[mount.body] = [];
      bodyMeshes[mount.body].push(m);
    }

    if (mount.meshes && mount.meshes.length > 0) {
      const subPromises = mount.meshes.map(async (sub) => {
        const geometry = await fetchSTL(botName, sub.file);
        if (!geometry) return;
        const mat = makeMaterial(materials[sub.material]);
        const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
        group.add(mesh);
        trackMountMesh(mesh);
      });
      await Promise.all(subPromises);
    } else {
      const geometry = await fetchSTL(botName, mount.mesh);
      if (!geometry) return;
      const mountColor = mount.color
        ? new THREE.Color(mount.color[0], mount.color[1], mount.color[2])
        : new THREE.Color(0x808080);
      const mat = new THREE.MeshPhysicalMaterial({
        color: mountColor,
        roughness: 0.5,
        metalness: 0.0,
      });
      const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
      group.add(mesh);
      trackMountMesh(mesh);
    }
  });

  // Load part meshes (fasteners, wires)
  // Track fastener index per body to match FastenerRef(body, index) from assembly ops
  const fastenerCountByBody: Record<string, number> = {};
  const partsList = manifest.parts ?? [];
  const partMeshPromises = partsList.map(async (part) => {
    if (!part.pos || !part.quat) return;
    const group = getBodyGroup(part.parent_body);

    const geometry = await fetchSTL(botName, part.mesh);
    if (!geometry) return;

    let mat: THREE.Material;
    if (part.category === 'fastener') {
      mat = new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.2 });
    } else if (part.color) {
      const c = new THREE.Color(part.color[0], part.color[1], part.color[2]);
      mat = new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5, emissive: c, emissiveIntensity: 0.15 });
    } else {
      mat = new THREE.MeshPhysicalMaterial({ color: 0x808080, roughness: 0.5, metalness: 0.0 });
    }

    const mesh = createPositionedMesh(geometry, mat, part.pos, part.quat);
    group.add(mesh);
    if (part.parent_body && !bodyMeshes[part.parent_body]) bodyMeshes[part.parent_body] = [];
    if (part.parent_body) bodyMeshes[part.parent_body].push(mesh);

    // Track part meshes for assembly scrubber visibility
    let partKey: string | null = null;
    if (part.category === 'fastener') {
      const idx = fastenerCountByBody[part.parent_body] ?? 0;
      fastenerCountByBody[part.parent_body] = idx + 1;
      partKey = `fastener:${part.parent_body}:${idx}`;
    } else if (part.category === 'wire') {
      partKey = `wire:${part.name}`;
    }
    if (partKey) {
      if (!partMeshes[partKey]) partMeshes[partKey] = [];
      partMeshes[partKey].push(mesh);
    }
  });

  await Promise.all([...bodyMeshPromises, ...mountMeshPromises, ...partMeshPromises]);

  // Frame camera on loaded geometry
  const box = new THREE.Box3();
  for (const group of Object.values(bodyGroups)) {
    box.expandByObject(group);
  }
  if (!box.isEmpty()) {
    viewport.frameOnBox(box);
  }

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
  // Finding click: ghost other bodies, fly camera to affected body
  // ---------------------------------------------------------------------------

  function onFindingClick(finding: DFMFindingData): void {
    const meshesForBody = bodyMeshes[finding.body];

    // Reset all body opacities
    for (const group of Object.values(bodyGroups)) {
      group.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
          if (mat.opacity !== undefined) {
            mat.transparent = false;
            mat.opacity = 1.0;
          }
        }
      });
    }

    if (meshesForBody && meshesForBody.length > 0) {
      // Ghost everything except the affected body
      for (const [name, group] of Object.entries(bodyGroups)) {
        const isTarget = name === finding.body;
        group.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mat = (child as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
            if (!isTarget) {
              mat.transparent = true;
              mat.opacity = 0.06;
            }
          }
        });
      }

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
      const allBox = new THREE.Box3();
      for (const group of Object.values(bodyGroups)) {
        allBox.expandByObject(group);
      }
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

    // Collect what is visible at this step by scanning ops 0..step
    const visibleBodies = new Set<string>();
    const visibleMounts = new Set<string>();
    const visibleParts = new Set<string>();

    for (let i = 0; i <= step; i++) {
      const op = assemblyOps[i];
      if (!op) continue;
      const t = op.target;

      if (op.action === 'insert') {
        if (t.type === 'component' && t.mount_label) {
          // Mount-level component (battery, Pi, camera, etc.)
          visibleMounts.add(`${t.body}:${t.mount_label}`);
        }
        // Body-level component (servo, horn) — always add the body
        visibleBodies.add(op.body);
      } else if (op.action === 'fasten' && t.type === 'fastener') {
        visibleParts.add(`fastener:${t.body}:${t.index}`);
      } else if (op.action === 'route_wire' && t.type === 'wire') {
        visibleParts.add(`wire:${t.label}`);
      }
    }

    // Structural bodies are always visible
    for (const body of manifest.bodies) {
      if (body.role !== 'component') {
        visibleBodies.add(body.name);
      }
    }

    // Apply body group visibility
    for (const [name, group] of Object.entries(bodyGroups)) {
      group.visible = visibleBodies.has(name);
    }

    // Apply mount mesh visibility
    for (const [key, meshes] of Object.entries(mountMeshes)) {
      const vis = visibleMounts.has(key);
      for (const m of meshes) m.visible = vis;
    }

    // Apply part mesh visibility
    for (const [key, meshes] of Object.entries(partMeshes)) {
      const vis = visibleParts.has(key);
      for (const m of meshes) m.visible = vis;
    }
  }

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
    } catch (err) {
      console.warn('[dfm] assembly-sequence error:', err);
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
      // Remove body groups from scene
      for (const group of Object.values(bodyGroups)) {
        viewport.scene.remove(group);
      }
    },
  };
}
