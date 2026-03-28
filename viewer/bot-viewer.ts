/**
 * MindSim Bot Viewer — MuJoCo-based bot visualization.
 *
 * Loads MuJoCo WASM, builds Three.js scene from the bot model,
 * and routes between Explore/Joint/Assembly/IK modes.
 *
 * Based on the monolithic viewer.js, extracted as an importable module
 * so viewer.js can act as a URL router.
 */

import * as THREE from 'three';
import { AssemblyMode } from './assembly-mode.ts';
import { BotScene } from './bot-scene.ts';
import { ExploreMode } from './explore-mode.ts';
import { FocusController } from './focus-controller.ts';
import { IKMode } from './ik-mode.ts';
import { JointMode } from './joint-mode.ts';
import { sync } from './scene-sync.ts';
import { SectionCutter } from './section-cutter.ts';
import { StressMode } from './stress-mode.ts';
import type { ViewerContext, ViewerMode as ViewerModeInterface } from './types.ts';
import { GEOM_GROUP_STRUCTURAL } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SIDE_PANEL_WIDTH = 320;
const TREE_PANEL_WIDTH = 280;

// ---------------------------------------------------------------------------
// Multi-material mesh enhancement
// ---------------------------------------------------------------------------

interface ManifestMesh {
  file: string;
  material: string;
}

interface ManifestMaterial {
  color: number[];
  metallic: number;
  roughness: number;
  opacity: number;
}

interface ManifestPart {
  id: string;
  name?: string;
  kind?: string;
  category?: string;
  mesh?: string;
  meshes?: ManifestMesh[];
  parent_body?: string;
  joint?: string;
  mount_label?: string;
  pos?: number[];
  quat?: number[];
  mass?: number;
  shapescript_component?: string;
  bus_type?: string;
}

/**
 * Replace single-color MuJoCo component meshes with per-material sub-meshes.
 *
 * For each component part that has a `meshes` array in the manifest, we:
 * 1. Load each per-material STL directly via fetch
 * 2. Create a MeshPhysicalMaterial with the material's visual properties
 * 3. Replace the existing MuJoCo geom mesh with a Group of sub-meshes
 */
async function enhanceMultiMaterialParts(
  manifest: any,
  bodies: Record<number, any>,
  botName: string,
  getMujocoName: (adrArray: any, index: number) => string,
  model: any,
): Promise<void> {
  const materials: Record<string, ManifestMaterial> = manifest.materials || {};

  // Multi-material meshes are now on mounts[] (mounted components).
  // Build a unified list that has the fields this function needs: id, parent_body, meshes.
  const mounts: any[] = manifest.mounts || [];
  const multiMaterialParts: ManifestPart[] = mounts
    .filter((m: any) => m.meshes && m.meshes.length > 0)
    .map(
      (m: any) =>
        ({
          id: `comp_${m.body}_${m.label}`,
          parent_body: m.body,
          meshes: m.meshes,
        }) as ManifestPart,
    );
  // Also check legacy parts[] for backward compatibility
  for (const p of (manifest.parts || []) as ManifestPart[]) {
    if (p.meshes && p.meshes.length > 0) {
      multiMaterialParts.push(p);
    }
  }
  if (multiMaterialParts.length === 0) return;

  // Build a name→bodyID lookup
  const nameToBody: Record<string, number> = {};
  for (let b = 0; b < model.nbody; b++) {
    const name = getMujocoName(model.name_bodyadr, b);
    if (name) nameToBody[name] = b;
  }

  // Build a geomName→geom mesh lookup within each body group
  const { STLLoader } = await import('three/addons/loaders/STLLoader.js');
  const stlLoader = new STLLoader();

  for (const part of multiMaterialParts) {
    // Find the parent body that contains this component's MuJoCo geom
    const parentBodyName = part.parent_body;
    if (!parentBodyName) {
      console.warn('[multi-mat] no parent_body for', part.id);
      continue;
    }

    const parentBodyId = nameToBody[parentBodyName];
    if (parentBodyId === undefined) {
      console.warn('[multi-mat] no body ID for', parentBodyName);
      continue;
    }
    const parentGroup = bodies[parentBodyId];
    if (!parentGroup) {
      console.warn('[multi-mat] no group for body', parentBodyId);
      continue;
    }

    // Find the MuJoCo geom mesh for this component (by geomName matching part.id)
    let existingMesh: any = null;
    const geomNames: string[] = [];
    parentGroup.traverse((child: any) => {
      if (child.isMesh) {
        geomNames.push(child.geomName ?? '(none)');
        if (child.geomName === part.id) {
          existingMesh = child;
        }
      }
    });
    if (!existingMesh) {
      console.warn('[multi-mat] no geom match for', part.id, '— available:', geomNames);
      continue;
    }
    console.log('[multi-mat] replacing', part.id, 'with', part.meshes!.length, 'sub-meshes');

    // Load per-material STLs in parallel
    const meshEntries = part.meshes!;
    const loadPromises = meshEntries.map(async (entry: ManifestMesh) => {
      const matDef = materials[entry.material];
      try {
        const resp = await fetch(`/api/bots/${botName}/meshes/${entry.file}`);
        if (!resp.ok) return null;
        const buf = await resp.arrayBuffer();
        const geometry = stlLoader.parse(buf);
        geometry.computeVertexNormals();

        const color = matDef
          ? new THREE.Color(matDef.color[0], matDef.color[1], matDef.color[2])
          : new THREE.Color(0.7, 0.7, 0.7);

        const mat = new THREE.MeshPhysicalMaterial({
          color,
          metalness: matDef?.metallic ?? 0.0,
          roughness: matDef?.roughness ?? 0.7,
          transparent: (matDef?.opacity ?? 1.0) < 1.0,
          opacity: matDef?.opacity ?? 1.0,
        });

        const mesh = new THREE.Mesh(geometry, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        return mesh;
      } catch {
        return null;
      }
    });

    const subMeshes = (await Promise.all(loadPromises)).filter(Boolean);
    if (subMeshes.length === 0) continue;

    // Tag original mesh so sync() keeps it hidden (replaced by multi-material)
    (existingMesh as any)._multiMaterialReplaced = true;
    existingMesh.visible = false;

    // Add sub-meshes to the parent group with the geom's position offset
    // but NOT its quaternion. The per-material STLs already have mount
    // rotation baked in (from _apply_mount_rotation in build_cad), same as
    // the original STL. The MuJoCo geom quaternion represents the same
    // rotation, so copying it would double-apply it. But the position offset
    // (geom center relative to body origin) is still needed.
    for (const subMesh of subMeshes) {
      subMesh.position.copy(existingMesh.position);
      subMesh.scale.copy(existingMesh.scale);
      // Tag sub-meshes with the same body/geom metadata
      (subMesh as any).bodyID = (existingMesh as any).bodyID;
      (subMesh as any).geomGroup = (existingMesh as any).geomGroup;
      (subMesh as any).geomName = `${part.id}_multi`;
      parentGroup.add(subMesh);
    }
  }
}

export interface BotViewerHandle {
  pause(): void;
  resume(): void;
}

export async function initBotViewer(botName: string): Promise<BotViewerHandle> {
  // ---------------------------------------------------------------------------
  // Globals shared across modes
  // ---------------------------------------------------------------------------
  /** @type {any} */ let mujoco: any;
  /** @type {any} */ let model: any;
  /** @type {any} */ let data: any;
  /** @type {Object<number, THREE.Group>} */ const bodies: Record<number, any> = {};
  let mujocoRoot: any;

  // Cached name decoding (allocated once after model load)
  let _namesArray: Uint8Array | null = null;
  const _textDecoder = new TextDecoder('utf-8');

  // Three.js — delegated to Viewport3D
  const container = document.getElementById('canvas-container');

  // Restore persisted tree panel width or use default
  const TREE_MIN_WIDTH = 200;
  const TREE_MAX_WIDTH = 500;
  let treePanelWidth = TREE_PANEL_WIDTH;
  try {
    const saved = localStorage.getItem('mindsim-tree-panel-width');
    if (saved) {
      const w = parseInt(saved, 10);
      if (w >= TREE_MIN_WIDTH && w <= TREE_MAX_WIDTH) treePanelWidth = w;
    }
  } catch {
    /* ignore */
  }

  function getTreePanelWidth() {
    const treePanel = document.getElementById('tree-panel');
    return treePanel && treePanel.style.display !== 'none' ? treePanelWidth : 0;
  }

  // Apply initial tree panel width
  const treePanelEl = document.getElementById('tree-panel');
  if (treePanelEl) treePanelEl.style.width = `${treePanelWidth}px`;

  // ── Resizable tree panel ──
  function initTreeResize() {
    const panel = document.getElementById('tree-panel');
    if (!panel) return;

    const handle = document.createElement('div');
    handle.className = 'tree-resize-handle';
    panel.appendChild(handle);

    let dragging = false;
    let startX = 0;
    let startWidth = 0;

    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      dragging = true;
      startX = e.clientX;
      startWidth = treePanelWidth;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const newWidth = Math.max(TREE_MIN_WIDTH, Math.min(TREE_MAX_WIDTH, startWidth + dx));
      treePanelWidth = newWidth;
      panel.style.width = `${newWidth}px`;
      updateCanvasLayout();
    });

    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      try {
        localStorage.setItem('mindsim-tree-panel-width', String(treePanelWidth));
      } catch {
        /* ignore */
      }
    });
  }
  initTreeResize();

  // Position container between tree panel and side panel
  container.style.left = `${getTreePanelWidth()}px`;
  container.style.right = `${SIDE_PANEL_WIDTH}px`;

  const viewport = new Viewport3D(container, {
    cameraType: 'orthographic',
    grid: true,
  });
  const scene = viewport.scene;
  const camera = viewport.camera;
  const renderer = viewport.renderer;
  const controls = viewport.controls;

  // Bot-viewer starts at a specific camera position
  camera.position.set(0.4, 0.45, 0.55);
  controls.target.set(0, 0.2, 0);
  controls.update();

  /** Update container bounds and let Viewport3D handle resize. */
  function updateCanvasLayout() {
    const left = getTreePanelWidth();
    container.style.left = `${left}px`;
    container.style.right = `${SIDE_PANEL_WIDTH}px`;
    viewport.resize();
  }

  // ---------------------------------------------------------------------------
  // Coordinate helpers — MuJoCo is Z-up, Three.js viewport is Z-up, no swizzle needed
  // ---------------------------------------------------------------------------
  function getPosition(buffer: any, index: number, target: THREE.Vector3) {
    return target.set(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
  }

  function getQuaternion(buffer: any, index: number, target: THREE.Quaternion) {
    // MuJoCo quaternion layout: [w, x, y, z]
    // Three.js Quaternion constructor: (x, y, z, w)
    return target.set(buffer[index * 4 + 1], buffer[index * 4 + 2], buffer[index * 4 + 3], buffer[index * 4 + 0]);
  }

  function toMujocoPos(v: any) {
    return v; // identity — same coordinate system
  }

  // ---------------------------------------------------------------------------
  // Load MuJoCo WASM
  // ---------------------------------------------------------------------------
  async function loadMuJoCo() {
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = 'Loading MuJoCo WASM...';

    const { default: load_mujoco } = await import(
      // @ts-expect-error — dynamic CDN import, no types available
      'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js'
    );
    mujoco = await load_mujoco();

    // Set up virtual filesystem
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
    mujoco.FS.mkdir('/working/meshes');

    loadingText.textContent = 'Loading bot model...';

    // Fetch bot.xml from API (on-demand generation from skeleton)
    const botApi = `/api/bots/${botName}`;
    const xmlText = await (await fetch(`${botApi}/bot.xml`)).text();
    mujoco.FS.writeFile('/working/bot.xml', xmlText);

    // Parse XML to find mesh files and fetch them in parallel from API
    const xmlDoc = new DOMParser().parseFromString(xmlText, 'text/xml');
    const meshFiles = [...xmlDoc.querySelectorAll('mesh[file]')].map((el) => el.getAttribute('file'));

    loadingText.textContent = `Loading ${meshFiles.length} meshes...`;
    await Promise.all(
      meshFiles.map(async (file) => {
        const buf = new Uint8Array(await (await fetch(`${botApi}/meshes/${file}`)).arrayBuffer());
        mujoco.FS.writeFile(`/working/meshes/${file}`, buf);
      }),
    );

    // Load model
    loadingText.textContent = 'Initializing simulation...';
    model = mujoco.MjModel.loadFromXML('/working/bot.xml');
    data = new mujoco.MjData(model);
    mujoco.mj_forward(model, data);

    // Cache name array once
    _namesArray = new Uint8Array(model.names);
  }

  // ---------------------------------------------------------------------------
  // Decode MuJoCo name strings (cached array, single TextDecoder)
  // ---------------------------------------------------------------------------
  function getMujocoName(adrArray: any, index: number) {
    const start = adrArray[index];
    let end = start;
    while (end < _namesArray.length && _namesArray[end] !== 0) end++;
    return _textDecoder.decode(_namesArray.subarray(start, end));
  }

  // ---------------------------------------------------------------------------
  // Build Three.js scene from MuJoCo model
  // ---------------------------------------------------------------------------
  function buildScene() {
    mujocoRoot = viewport.addGroup('mujoco');
    mujocoRoot.name = 'MuJoCo Root';

    const meshCache: Record<number, THREE.BufferGeometry> = {};

    for (let g = 0; g < model.ngeom; g++) {
      const b = model.geom_bodyid[g];
      const type = model.geom_type[g];
      const size = [model.geom_size[g * 3 + 0], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];

      if (!(b in bodies)) {
        bodies[b] = new THREE.Group();
        bodies[b].name = getMujocoName(model.name_bodyadr, b);
        bodies[b].bodyID = b;
        bodies[b].has_custom_mesh = false;
      }

      let geometry: THREE.BufferGeometry | undefined;
      const mjtGeom = mujoco.mjtGeom;
      if (type === mjtGeom.mjGEOM_PLANE.value) {
        continue;
      } else if (type === mjtGeom.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0], 16, 16);
      } else if (type === mjtGeom.mjGEOM_CAPSULE.value) {
        // MuJoCo capsule extends along Z; Three.js CapsuleGeometry extends along Y.
        // Rotate -90° around X to align Y→Z.
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 12, 12);
        geometry.rotateX(-Math.PI / 2);
      } else if (type === mjtGeom.mjGEOM_CYLINDER.value) {
        // MuJoCo cylinder extends along Z; Three.js CylinderGeometry extends along Y.
        // Rotate -90° around X to align Y→Z.
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0, 16);
        geometry.rotateX(-Math.PI / 2);
      } else if (type === mjtGeom.mjGEOM_BOX.value) {
        // MuJoCo box size = [half_x, half_y, half_z]; Three.js BoxGeometry(x, y, z)
        geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
      } else if (type === mjtGeom.mjGEOM_MESH.value) {
        const meshID = model.geom_dataid[g];
        if (!(meshID in meshCache)) {
          geometry = new THREE.BufferGeometry();
          const vertBuf = model.mesh_vert.subarray(
            model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3,
          );
          // No vertex swizzle — MuJoCo meshes are Z-up, matching our viewport
          const faceBuf = model.mesh_face.subarray(
            model.mesh_faceadr[meshID] * 3,
            (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3,
          );
          geometry.setAttribute('position', new THREE.BufferAttribute(vertBuf, 3));
          geometry.setIndex(Array.from(faceBuf));
          geometry.computeVertexNormals();
          meshCache[meshID] = geometry;
        } else {
          geometry = meshCache[meshID];
        }
        bodies[b].has_custom_mesh = true;
      } else {
        geometry = new THREE.SphereGeometry(size[0] || 0.005, 8, 8);
      }

      const color = [
        model.geom_rgba[g * 4 + 0],
        model.geom_rgba[g * 4 + 1],
        model.geom_rgba[g * 4 + 2],
        model.geom_rgba[g * 4 + 3],
      ];
      const mat = new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3],
        roughness: 0.6,
        metalness: 0.1,
      });

      const mesh = new THREE.Mesh(geometry, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      (mesh as any).bodyID = b;
      (mesh as any).geomID = g;
      (mesh as any).geomGroup = model.geom_group[g];
      (mesh as any).geomName = getMujocoName(model.name_geomadr, g);

      bodies[b].add(mesh);

      getPosition(model.geom_pos, g, mesh.position);
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }

    // Assign distinct Blueprint.js palette colors to mesh-type bodies.
    // Only override the default gray (0.9, 0.9, 0.9); leave detail geoms alone.
    const bodyColors = [
      null, // body 0 = world, skip
      [0.808, 0.851, 0.878], // base — BP_LIGHT_GRAY1 (#CED9E0)
      [0.094, 0.133, 0.157], // left_rim — BP_DARK_GRAY1 (#182026)
      [0.094, 0.133, 0.157], // right_rim — BP_DARK_GRAY1 (#182026)
      [0.169, 0.584, 0.839], // turntable — BP_BLUE4 (#2B95D6)
      [0.655, 0.761, 0.831], // upper_arm — BP_GRAY4 (#A7B6C2)
      [0.541, 0.608, 0.659], // forearm — BP_GRAY3 (#8A9BA8)
      [0.851, 0.62, 0.043], // hand — BP_GOLD3 (#D99E0B)
    ];
    for (const [b, group] of Object.entries(bodies)) {
      const bi = parseInt(b, 10);
      if (bi > 0 && bi < bodyColors.length && bodyColors[bi] && group.has_custom_mesh) {
        group.traverse((child) => {
          if (child.isMesh && child.geomGroup === GEOM_GROUP_STRUCTURAL) {
            const [r, g, bl] = bodyColors[bi];
            child.material.color.setRGB(r, g, bl);
          }
        });
      }
    }

    // Cell-shaded edge outlines — sharp edges only (threshold angle ~30°)
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.6,
    });
    for (const [, group] of Object.entries(bodies)) {
      const meshesToEdge = [];
      group.traverse((child) => {
        if (child.isMesh && child.geometry && child.geomGroup === GEOM_GROUP_STRUCTURAL) meshesToEdge.push(child);
      });
      for (const mesh of meshesToEdge) {
        const edges = new THREE.EdgesGeometry(mesh.geometry, 28);
        const lines = new THREE.LineSegments(edges, edgeMaterial);
        lines.position.copy(mesh.position);
        lines.quaternion.copy(mesh.quaternion);
        lines.scale.copy(mesh.scale);
        lines.raycast = () => {}; // non-interactive
        group.add(lines);
      }
    }

    // Parent all bodies to root (flat — transforms come from xpos/xquat)
    for (let b = 0; b < model.nbody; b++) {
      if (!bodies[b]) {
        bodies[b] = new THREE.Group();
        bodies[b].name = `body_${b}`;
        bodies[b].bodyID = b;
        bodies[b].has_custom_mesh = false;
      }
      if (b === 0) {
        mujocoRoot.add(bodies[b]);
      } else {
        bodies[0].add(bodies[b]);
      }
    }

    syncTransforms();
  }

  const _syncVec = new THREE.Vector3();
  const _syncQuat = new THREE.Quaternion();

  function syncTransforms() {
    const tmpVec = _syncVec;
    const tmpQuat = _syncQuat;
    for (let b = 0; b < model.nbody; b++) {
      if (bodies[b]) {
        getPosition(data.xpos, b, tmpVec);
        getQuaternion(data.xquat, b, tmpQuat);
        bodies[b].position.copy(tmpVec);
        bodies[b].quaternion.copy(tmpQuat);
        bodies[b].updateWorldMatrix();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Fetch viewer manifest
  // ---------------------------------------------------------------------------
  async function fetchManifest() {
    try {
      const resp = await fetch(`../bots/${botName}/viewer_manifest.json`);
      if (resp.ok) return await resp.json();
    } catch {
      /* fall through */
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Mode management
  // ---------------------------------------------------------------------------
  let currentMode: ViewerModeInterface | null = null;
  let currentModeName: string | null = null;
  const modes: Record<string, ViewerModeInterface> = {};

  function switchMode(modeName: string) {
    if (!(modeName in modes)) return;
    if (currentMode) currentMode.deactivate();
    currentMode = modes[modeName];
    currentModeName = modeName;
    currentMode.activate();

    // Refit canvas to the visible area between panels
    requestAnimationFrame(() => updateCanvasLayout());

    document.querySelectorAll('#sim-mode-tabs .btn-ghost').forEach((tab) => {
      tab.classList.toggle('active', (tab as HTMLElement).dataset.mode === modeName);
    });
  }

  // ---------------------------------------------------------------------------
  // Main init
  // ---------------------------------------------------------------------------
  try {
    await loadMuJoCo();
    buildScene();

    const manifest = await fetchManifest();

    // Build body name list for the data model
    const bodyNames: string[] = [];
    for (let b = 0; b < model.nbody; b++) {
      bodyNames.push(getMujocoName(model.name_bodyadr, b) || `body_${b}`);
    }
    const botScene = new BotScene(model.nbody, bodyNames);

    /** Apply BotScene state to Three.js. Call after every BotScene mutation. */
    function syncScene() {
      sync(botScene, bodies);
    }

    const ctx: ViewerContext = {
      mujoco,
      model,
      data,
      bodies,
      mujocoRoot,
      scene,
      camera,
      renderer,
      controls,
      viewport,
      syncTransforms,
      getPosition,
      getQuaternion,
      toMujocoPos,
      getMujocoName,
      botName,
      botScene,
      syncScene,
    };

    // Multi-material: replace single-color component meshes with per-material sub-meshes
    if (manifest) {
      await enhanceMultiMaterialParts(manifest, bodies, botName, getMujocoName, model);
      modes.explore = new ExploreMode(ctx, manifest);
    }
    modes.joint = new JointMode(ctx);
    modes.assembly = new AssemblyMode(ctx);
    modes.ik = new IKMode(ctx);
    modes.stress = new StressMode(ctx);

    // Build sim mode tabs dynamically (the #mode-tabs div is shared with Design/Sim
    // tab buttons, so we create a separate container for Explore/Stress/etc.)
    let simModeTabs = document.getElementById('sim-mode-tabs');
    if (!simModeTabs) {
      simModeTabs = document.createElement('div');
      simModeTabs.id = 'sim-mode-tabs';
      simModeTabs.className = 'nav-right';
      simModeTabs.style.cssText = 'display:flex;gap:2px;';
      const topBar = document.getElementById('top-bar');
      // Insert before the Design/Sim tabs
      const modeTabsEl = document.getElementById('mode-tabs');
      if (modeTabsEl) {
        topBar.insertBefore(simModeTabs, modeTabsEl);
      } else {
        topBar.appendChild(simModeTabs);
      }
    }
    simModeTabs.innerHTML = '';
    const simModeNames = ['explore', 'stress', 'joint', 'assembly', 'ik'];
    const simModeLabels: Record<string, string> = {
      explore: 'Explore',
      stress: 'Stress',
      joint: 'Joints',
      assembly: 'Assembly',
      ik: 'IK',
    };
    for (const name of simModeNames) {
      const btn = document.createElement('button');
      btn.className = 'btn-ghost';
      btn.dataset.mode = name;
      btn.textContent = simModeLabels[name] || name;
      btn.addEventListener('click', () => switchMode(name));
      simModeTabs.appendChild(btn);
    }

    // Auto-focus camera on the bot before showing the scene
    const initialFocus = new FocusController(ctx);
    initialFocus.focusOnAll(0.8);

    // Section cutter with per-body cap colors
    const sectionCutter = new SectionCutter(scene, renderer);
    sectionCutter.setMeshProvider(() => {
      const meshes: any[] = [];
      for (const bodyId of botScene.visibleBodyIds()) {
        const group = bodies[bodyId];
        if (!group) continue;
        group.traverse((ch: any) => {
          if (ch.isMesh && ch.geomGroup === GEOM_GROUP_STRUCTURAL) meshes.push(ch);
        });
      }
      return meshes;
    });
    sectionCutter.setCapColorFn((mesh) => {
      const c = mesh.material.color;
      return new THREE.Color().copy(c);
    });
    sectionCutter.bindUI({
      toggle: document.getElementById('bot-section-toggle'),
      controls: document.getElementById('bot-section-controls'),
      axisButtons: document.querySelectorAll('#bot-section-controls [data-section-axis]'),
      slider: document.getElementById('bot-section-slider'),
      flipBtn: document.getElementById('bot-section-flip'),
      keyTarget: window,
    });
    // Show bot tools when in bot viewer
    document.getElementById('bot-tools-group').style.display = '';

    switchMode(manifest ? 'explore' : 'joint');
    document.getElementById('loading').style.display = 'none';

    let paused = false;

    viewport.animate(() => {
      if (paused) return;
      initialFocus.update(); // drive initial camera animation
      if (currentMode?.update) currentMode.update();
    });

    window.addEventListener('resize', () => updateCanvasLayout());

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;
      if (e.key === '1' && currentModeName === 'explore' && modes.explore) {
        (modes.explore as ExploreMode).refocusCurrent();
      }
    });

    return {
      pause() {
        paused = true;
        // Deactivate current mode to clean up event listeners
        if (currentMode) currentMode.deactivate();
      },
      resume() {
        paused = false;
        // Re-activate current mode
        if (currentMode) currentMode.activate();
        updateCanvasLayout();
      },
    };
  } catch (err) {
    console.error('Failed to initialize viewer:', err);
    document.getElementById('loading-text').textContent = `Error: ${err.message}`;
    return { pause() {}, resume() {} };
  }
}
