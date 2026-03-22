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
import { JointMode } from './joint-mode.js';
import { AssemblyMode } from './assembly-mode.js';
import { IKMode } from './ik-mode.js';
import { ExploreMode } from './explore-mode.js';
import { FocusController } from './focus-controller.js';
import { Viewport3D } from './viewport3d.js';
import { GEOM_GROUP_STRUCTURAL } from './utils.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SIDE_PANEL_WIDTH = 320;
const TREE_PANEL_WIDTH = 280;

export async function initBotViewer(botName) {
  // ---------------------------------------------------------------------------
  // Globals shared across modes
  // ---------------------------------------------------------------------------
  /** @type {any} */ let mujoco;
  /** @type {any} */ let model;
  /** @type {any} */ let data;
  /** @type {Object<number, THREE.Group>} */ const bodies = {};
  let mujocoRoot;

  // Cached name decoding (allocated once after model load)
  let _namesArray = null;
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
  } catch { /* ignore */ }

  function getTreePanelWidth() {
    const treePanel = document.getElementById('tree-panel');
    return (treePanel && treePanel.style.display !== 'none') ? treePanelWidth : 0;
  }

  // Apply initial tree panel width
  const treePanelEl = document.getElementById('tree-panel');
  if (treePanelEl) treePanelEl.style.width = treePanelWidth + 'px';

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
      panel.style.width = newWidth + 'px';
      updateCanvasLayout();
    });

    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      try { localStorage.setItem('mindsim-tree-panel-width', String(treePanelWidth)); } catch { /* ignore */ }
    });
  }
  initTreeResize();

  // Position container between tree panel and side panel
  container.style.left = getTreePanelWidth() + 'px';
  container.style.right = SIDE_PANEL_WIDTH + 'px';

  const viewport = new Viewport3D(container, {
    cameraType: 'perspective',
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
    container.style.left = left + 'px';
    container.style.right = SIDE_PANEL_WIDTH + 'px';
    viewport.resize();
  }

  // ---------------------------------------------------------------------------
  // Coordinate swizzle helpers (MuJoCo Y-up → Three.js Y-up with Z-flip)
  // ---------------------------------------------------------------------------
  function getPosition(buffer, index, target) {
    return target.set(
      buffer[index * 3 + 0],
      buffer[index * 3 + 2],
      -buffer[index * 3 + 1]
    );
  }

  function getQuaternion(buffer, index, target) {
    return target.set(
      -buffer[index * 4 + 1],
      -buffer[index * 4 + 3],
      buffer[index * 4 + 2],
      -buffer[index * 4 + 0]
    );
  }

  function toMujocoPos(v) {
    return v.set(v.x, -v.z, v.y);
  }

  // ---------------------------------------------------------------------------
  // Load MuJoCo WASM
  // ---------------------------------------------------------------------------
  async function loadMuJoCo() {
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = 'Loading MuJoCo WASM...';

    const { default: load_mujoco } = await import(
      'https://cdn.jsdelivr.net/npm/mujoco-js@0.0.7/dist/mujoco_wasm.js'
    );
    mujoco = await load_mujoco();

    // Set up virtual filesystem
    mujoco.FS.mkdir('/working');
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
    mujoco.FS.mkdir('/working/meshes');

    loadingText.textContent = 'Loading bot model...';

    // Fetch bot.xml
    const botDir = `../bots/${botName}`;
    const xmlText = await (await fetch(`${botDir}/bot.xml`)).text();
    mujoco.FS.writeFile('/working/bot.xml', xmlText);

    // Parse XML to find mesh files and fetch them in parallel
    const xmlDoc = new DOMParser().parseFromString(xmlText, 'text/xml');
    const meshFiles = [...xmlDoc.querySelectorAll('mesh[file]')].map(el => el.getAttribute('file'));

    loadingText.textContent = `Loading ${meshFiles.length} meshes...`;
    await Promise.all(meshFiles.map(async (file) => {
      const buf = new Uint8Array(await (await fetch(`${botDir}/meshes/${file}`)).arrayBuffer());
      mujoco.FS.writeFile(`/working/meshes/${file}`, buf);
    }));

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
  function getMujocoName(adrArray, index) {
    const start = adrArray[index];
    let end = start;
    while (end < _namesArray.length && _namesArray[end] !== 0) end++;
    return _textDecoder.decode(_namesArray.subarray(start, end));
  }

  // ---------------------------------------------------------------------------
  // Build Three.js scene from MuJoCo model
  // ---------------------------------------------------------------------------
  function buildScene() {
    mujocoRoot = new THREE.Group();
    mujocoRoot.name = 'MuJoCo Root';
    scene.add(mujocoRoot);

    const meshCache = {};

    for (let g = 0; g < model.ngeom; g++) {
      const b = model.geom_bodyid[g];
      const type = model.geom_type[g];
      const size = [
        model.geom_size[g * 3 + 0],
        model.geom_size[g * 3 + 1],
        model.geom_size[g * 3 + 2],
      ];

      if (!(b in bodies)) {
        bodies[b] = new THREE.Group();
        bodies[b].name = getMujocoName(model.name_bodyadr, b);
        bodies[b].bodyID = b;
        bodies[b].has_custom_mesh = false;
      }

      let geometry;
      const mjtGeom = mujoco.mjtGeom;
      if (type === mjtGeom.mjGEOM_PLANE.value) {
        continue;
      } else if (type === mjtGeom.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0], 16, 16);
      } else if (type === mjtGeom.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 12, 12);
      } else if (type === mjtGeom.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0, 16);
      } else if (type === mjtGeom.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
      } else if (type === mjtGeom.mjGEOM_MESH.value) {
        const meshID = model.geom_dataid[g];
        if (!(meshID in meshCache)) {
          geometry = new THREE.BufferGeometry();
          const vertBuf = model.mesh_vert.subarray(
            model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID] + model.mesh_vertnum[meshID]) * 3
          );
          for (let v = 0; v < vertBuf.length; v += 3) {
            const tmp = vertBuf[v + 1];
            vertBuf[v + 1] = vertBuf[v + 2];
            vertBuf[v + 2] = -tmp;
          }
          const faceBuf = model.mesh_face.subarray(
            model.mesh_faceadr[meshID] * 3,
            (model.mesh_faceadr[meshID] + model.mesh_facenum[meshID]) * 3
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
      mesh.bodyID = b;
      mesh.geomID = g;
      mesh.geomGroup = model.geom_group[g];
      mesh.geomName = getMujocoName(model.name_geomadr, g);

      bodies[b].add(mesh);

      getPosition(model.geom_pos, g, mesh.position);
      getQuaternion(model.geom_quat, g, mesh.quaternion);
    }

    // Assign distinct Blueprint.js palette colors to mesh-type bodies.
    // Only override the default gray (0.9, 0.9, 0.9); leave detail geoms alone.
    const bodyColors = [
      null,                              // body 0 = world, skip
      [0.808, 0.851, 0.878],            // base — BP_LIGHT_GRAY1 (#CED9E0)
      [0.094, 0.133, 0.157],            // left_rim — BP_DARK_GRAY1 (#182026)
      [0.094, 0.133, 0.157],            // right_rim — BP_DARK_GRAY1 (#182026)
      [0.169, 0.584, 0.839],            // turntable — BP_BLUE4 (#2B95D6)
      [0.655, 0.761, 0.831],            // upper_arm — BP_GRAY4 (#A7B6C2)
      [0.541, 0.608, 0.659],            // forearm — BP_GRAY3 (#8A9BA8)
      [0.851, 0.620, 0.043],            // hand — BP_GOLD3 (#D99E0B)
    ];
    for (const [b, group] of Object.entries(bodies)) {
      const bi = parseInt(b);
      if (bi > 0 && bi < bodyColors.length && bodyColors[bi] && group.has_custom_mesh) {
        group.traverse(child => {
          if (child.isMesh && child.geomGroup === GEOM_GROUP_STRUCTURAL) {
            const [r, g, bl] = bodyColors[bi];
            child.material.color.setRGB(r, g, bl);
          }
        });
      }
    }

    // Cell-shaded edge outlines — sharp edges only (threshold angle ~30°)
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x000000, transparent: true, opacity: 0.6,
    });
    for (const [, group] of Object.entries(bodies)) {
      const meshesToEdge = [];
      group.traverse(child => {
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
    } catch { /* fall through */ }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Mode management
  // ---------------------------------------------------------------------------
  let currentMode = null;
  let currentModeName = null;
  const modes = {};

  function switchMode(modeName) {
    if (!(modeName in modes)) return;
    if (currentMode) currentMode.deactivate();
    currentMode = modes[modeName];
    currentModeName = modeName;
    currentMode.activate();

    // Refit canvas to the visible area between panels
    requestAnimationFrame(() => updateCanvasLayout());

    document.querySelectorAll('#mode-tabs .btn-ghost').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.mode === modeName);
    });
  }

  // ---------------------------------------------------------------------------
  // Main init
  // ---------------------------------------------------------------------------
  try {
    await loadMuJoCo();
    buildScene();

    const manifest = await fetchManifest();

    const ctx = {
      mujoco, model, data, bodies, mujocoRoot, scene, camera, renderer, controls, viewport,
      syncTransforms, getPosition, getQuaternion, toMujocoPos, getMujocoName,
      botName,
    };

    if (manifest) {
      modes.explore = new ExploreMode(ctx, manifest);
    }
    modes.joint = new JointMode(ctx);
    modes.assembly = new AssemblyMode(ctx);
    modes.ik = new IKMode(ctx);

    document.querySelectorAll('#mode-tabs .btn-ghost').forEach(tab => {
      tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });

    // Auto-focus camera on the bot before showing the scene
    const initialFocus = new FocusController(ctx);
    initialFocus.focusOnAll(0.8);

    switchMode(manifest ? 'explore' : 'joint');
    document.getElementById('loading').style.display = 'none';

    viewport.animate(() => {
      initialFocus.update();  // drive initial camera animation
      if (currentMode && currentMode.update) currentMode.update();
    });

    window.addEventListener('resize', () => updateCanvasLayout());

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1' && currentModeName === 'explore' && modes.explore) {
        modes.explore.refocusCurrent();
      }
    });

  } catch (err) {
    console.error('Failed to initialize viewer:', err);
    document.getElementById('loading-text').textContent = `Error: ${err.message}`;
  }
}
