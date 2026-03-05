/**
 * MindSim Bot Viewer — Main entry point.
 *
 * Loads MuJoCo WASM, initializes Three.js scene, routes between modes.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { JointMode } from './joint-mode.js';
import { AssemblyMode } from './assembly-mode.js';
import { IKMode } from './ik-mode.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SIDE_PANEL_WIDTH = 320;

// ---------------------------------------------------------------------------
// URL params
// ---------------------------------------------------------------------------
const params = new URLSearchParams(window.location.search);
const botName = params.get('bot') || 'wheeler_arm';
document.getElementById('bot-name').textContent = botName;

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

// Three.js
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(
  45, (window.innerWidth - SIDE_PANEL_WIDTH) / window.innerHeight, 0.001, 100
);
camera.position.set(0.4, 0.45, 0.55);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.2, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 4, 3);
dirLight.castShadow = true;
scene.add(dirLight);

// Ground grid
scene.add(new THREE.GridHelper(2, 40, 0x333355, 0x222244));

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
    if (model.geom_group[g] >= 3) continue;

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

  // Assign distinct colors to mesh-type bodies for visual differentiation.
  // Only override the default gray (0.9, 0.9, 0.9); leave detail geoms alone.
  const bodyColors = [
    null,              // body 0 = world, skip
    [0.85, 0.85, 0.88], // base — light steel
    [0.35, 0.35, 0.40], // left_rim — dark gray
    [0.35, 0.35, 0.40], // right_rim — dark gray
    [0.55, 0.65, 0.85], // turntable — blue-gray
    [0.75, 0.80, 0.90], // upper_arm — light blue
    [0.65, 0.75, 0.85], // forearm — medium blue
    [0.90, 0.85, 0.75], // hand — warm beige
  ];
  for (const [b, group] of Object.entries(bodies)) {
    const bi = parseInt(b);
    if (bi > 0 && bi < bodyColors.length && bodyColors[bi] && group.has_custom_mesh) {
      group.traverse(child => {
        if (child.isMesh && child.geomGroup === 0) {
          const [r, g, bl] = bodyColors[bi];
          child.material.color.setRGB(r, g, bl);
        }
      });
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

function syncTransforms() {
  const tmpVec = new THREE.Vector3();
  const tmpQuat = new THREE.Quaternion();
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
// Mode management
// ---------------------------------------------------------------------------
let currentMode = null;
const modes = {};

function switchMode(modeName) {
  if (!(modeName in modes)) return;
  if (currentMode) currentMode.deactivate();
  currentMode = modes[modeName];
  currentMode.activate();

  document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.mode === modeName);
  });
}

// ---------------------------------------------------------------------------
// Main init
// ---------------------------------------------------------------------------
async function main() {
  try {
    await loadMuJoCo();
    buildScene();

    const ctx = {
      mujoco, model, data, bodies, mujocoRoot, scene, camera, renderer, controls,
      syncTransforms, getPosition, getQuaternion, toMujocoPos, getMujocoName,
      botName,
    };

    modes.joint = new JointMode(ctx);
    modes.assembly = new AssemblyMode(ctx);
    modes.ik = new IKMode(ctx);

    document.querySelectorAll('.mode-tab').forEach(tab => {
      tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });

    switchMode('joint');
    document.getElementById('loading').style.display = 'none';

    function animate() {
      controls.update();
      if (currentMode && currentMode.update) currentMode.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = (window.innerWidth - SIDE_PANEL_WIDTH) / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

  } catch (err) {
    console.error('Failed to initialize viewer:', err);
    document.getElementById('loading-text').textContent = `Error: ${err.message}`;
  }
}

main();
