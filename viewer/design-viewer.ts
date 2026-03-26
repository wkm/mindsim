/**
 * Design Viewer — loads meshes from the API, positions them using manifest
 * data, and wires up the SceneTree for per-mesh visibility control.
 *
 * Entry point: initDesignViewer(botName, viewport, treePanelEl)
 */

import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { buildSceneTree } from './build-scene-tree.ts';
import { ComponentTree } from './component-tree.ts';
import { DesignScene, NodeKind } from './design-scene.ts';
import type { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface DesignViewerContext {
  scene: DesignScene;
  tree: ComponentTree;
  viewport: Viewport3D;
  syncVisibility(): void;
}

// ---------------------------------------------------------------------------
// Manifest sub-types (minimal, mirrors Python emit/viewer.py output)
// ---------------------------------------------------------------------------

interface ManifestMaterial {
  color: [number, number, number];
  metallic: number;
  roughness: number;
  opacity: number;
}

interface ManifestBody {
  name: string;
  mesh: string;
  kind: string;
  parent: string | null;
  pos: number[];
  quat: number[]; // wxyz
  color?: [number, number, number];
}

interface ManifestSubMesh {
  file: string;
  material: string;
}

interface ManifestPart {
  id: string;
  name: string;
  category: string;
  parent_body: string;
  joint?: string;
  mount_label?: string;
  mesh: string;
  pos: number[];
  quat: number[]; // wxyz
  meshes?: ManifestSubMesh[];
}

interface ManifestDesignLayer {
  kind: string;
  mesh: string;
  parent_body: string;
}

interface ManifestJoint {
  name: string;
  parent_body: string;
  child_body: string;
  design_layers?: ManifestDesignLayer[];
}

interface ViewerManifest {
  bot_name: string;
  bodies: ManifestBody[];
  joints: ManifestJoint[];
  parts?: ManifestPart[];
  materials?: Record<string, ManifestMaterial>;
  assemblies?: unknown[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const stlLoader = new STLLoader();

/** Convert manifest wxyz quaternion to Three.js xyzw quaternion. */
function manifestQuatToThree(q: number[]): THREE.Quaternion {
  // Manifest: [w, x, y, z] → Three.js: new Quaternion(x, y, z, w)
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

/** Create a MeshPhysicalMaterial from a manifest material definition. */
function makeMaterial(matDef: ManifestMaterial | undefined): THREE.MeshPhysicalMaterial {
  if (!matDef) {
    return new THREE.MeshPhysicalMaterial({ color: 0xb0b0b0, roughness: 0.7 });
  }
  const color = new THREE.Color(matDef.color[0], matDef.color[1], matDef.color[2]);
  return new THREE.MeshPhysicalMaterial({
    color,
    metalness: matDef.metallic,
    roughness: matDef.roughness,
    transparent: matDef.opacity < 1.0,
    opacity: matDef.opacity,
  });
}

/** Fetch an STL and parse it into a BufferGeometry with vertex normals. */
async function fetchSTL(botName: string, meshFile: string): Promise<THREE.BufferGeometry | null> {
  try {
    const resp = await fetch(`/api/bots/${botName}/meshes/${meshFile}`);
    if (!resp.ok) {
      console.warn(`[design-viewer] failed to fetch ${meshFile}: ${resp.status}`);
      return null;
    }
    const buf = await resp.arrayBuffer();
    const geometry = stlLoader.parse(buf);
    geometry.computeVertexNormals();
    return geometry;
  } catch (err) {
    console.warn(`[design-viewer] error loading ${meshFile}:`, err);
    return null;
  }
}

/** Create a positioned mesh from geometry, material, pos, and quat. */
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
  return mesh;
}

// ---------------------------------------------------------------------------
// Layer lazy-loading state
// ---------------------------------------------------------------------------

/** Track which layer nodes have been loaded (or are loading). */
const layerLoadState = new Map<string, 'loading' | 'loaded'>();

/** Lazy-load a design layer mesh on first visibility toggle. */
async function lazyLoadLayerMesh(
  nodeId: string,
  manifest: ViewerManifest,
  designScene: DesignScene,
  viewport: Viewport3D,
  botName: string,
  bodyGroups: Record<string, THREE.Group>,
): Promise<void> {
  if (layerLoadState.has(nodeId)) return;
  layerLoadState.set(nodeId, 'loading');

  // Parse layer nodeId: "layer:{jointName}:{kind}"
  const parts = nodeId.split(':');
  if (parts.length < 3) return;
  const jointName = parts[1];
  const layerKind = parts[2];

  // Find the matching design layer in the manifest
  const joint = manifest.joints.find((j) => j.name === jointName);
  if (!joint?.design_layers) return;

  const layer = joint.design_layers.find((dl) => dl.kind === layerKind);
  if (!layer) return;

  // Find parent body for positioning
  const parentBody = manifest.bodies.find((b) => b.name === layer.parent_body);
  const pos = parentBody?.pos ?? [0, 0, 0];
  const quat = parentBody?.quat ?? [1, 0, 0, 0];

  // Layer-specific semi-transparent materials
  const layerColors: Record<string, number> = {
    bracket: 0x4488cc,
    coupler: 0x44cc88,
    clearance: 0xcc4444,
    insertion: 0xccaa44,
  };

  const geometry = await fetchSTL(botName, layer.mesh);
  if (!geometry) {
    layerLoadState.delete(nodeId);
    return;
  }

  const mat = new THREE.MeshPhysicalMaterial({
    color: layerColors[layerKind] ?? 0x888888,
    transparent: true,
    opacity: 0.35,
    roughness: 0.6,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const mesh = createPositionedMesh(geometry, mat, pos, quat);
  const group = bodyGroups[layer.parent_body] ?? Object.values(bodyGroups)[0];
  if (group) {
    group.add(mesh);
  } else {
    viewport.scene.add(mesh);
  }
  designScene.registerMesh(nodeId, mesh);
  layerLoadState.set(nodeId, 'loaded');
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initDesignViewer(
  botName: string,
  viewport: Viewport3D,
  treePanelEl: HTMLElement,
): Promise<DesignViewerContext> {
  // Step 1: Fetch manifest
  const resp = await fetch(`/api/bots/${botName}/viewer_manifest`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch viewer manifest: ${resp.status}`);
  }
  const manifest: ViewerManifest = await resp.json();
  const materials = manifest.materials ?? {};

  // Step 2: Build SceneTree from manifest
  const sceneTree = buildSceneTree(manifest as any);
  const designScene = new DesignScene(sceneTree);

  // Step 3: Create a viewport group per body so Viewport3D tracks them for
  // bounding box computation, section cap generation, and raycasting.
  const bodyGroups: Record<string, THREE.Group> = {};
  for (const body of manifest.bodies) {
    bodyGroups[body.name] = viewport.addGroup(`body:${body.name}`);
  }

  // Helper: find the group for a given body name (falls back to first group)
  function getBodyGroup(bodyName: string): THREE.Group {
    return bodyGroups[bodyName] ?? Object.values(bodyGroups)[0];
  }

  // Step 4: Load body meshes
  const bodyMeshPromises = manifest.bodies.map(async (body) => {
    const geometry = await fetchSTL(botName, body.mesh);
    if (!geometry) return;

    // Use body color or default gray for fabricated bodies
    const bodyColor = body.color
      ? new THREE.Color(body.color[0], body.color[1], body.color[2])
      : new THREE.Color(0.85, 0.85, 0.85);
    const mat = new THREE.MeshPhysicalMaterial({
      color: bodyColor,
      roughness: 0.6,
      metalness: 0.0,
    });

    const mesh = createPositionedMesh(geometry, mat, body.pos, body.quat);
    const nodeId = `body:${body.name}`;
    getBodyGroup(body.name).add(mesh);
    designScene.registerMesh(nodeId, mesh);
  });

  // Step 5: Load component part meshes
  const partsList = manifest.parts ?? [];
  const partMeshPromises = partsList.map(async (part) => {
    // Skip parts without positioning data (e.g., wire segments)
    if (!part.pos || !part.quat) return;

    // Determine the tree node ID — mirrors build-scene-tree.ts convention
    const nodeId = resolvePartNodeId(part);
    const group = getBodyGroup(part.parent_body);

    if (part.meshes && part.meshes.length > 0) {
      // Multi-material part: load each sub-mesh with its own material
      const subPromises = part.meshes.map(async (sub) => {
        const geometry = await fetchSTL(botName, sub.file);
        if (!geometry) return;

        const matDef = materials[sub.material];
        const mat = makeMaterial(matDef);
        const mesh = createPositionedMesh(geometry, mat, part.pos, part.quat);
        group.add(mesh);
        designScene.registerMesh(nodeId, mesh);
      });
      await Promise.all(subPromises);
    } else {
      // Single-mesh part
      const geometry = await fetchSTL(botName, part.mesh);
      if (!geometry) return;

      // Use default gray for parts — the manifest materials dict is keyed
      // by material name, not part name, so parts without meshes[] use a
      // generic appearance.
      const mat = new THREE.MeshPhysicalMaterial({
        color: 0x333333,
        roughness: 0.5,
        metalness: 0.1,
      });
      const mesh = createPositionedMesh(geometry, mat, part.pos, part.quat);
      group.add(mesh);
      designScene.registerMesh(nodeId, mesh);
    }
  });

  // Wait for all eager meshes
  await Promise.all([...bodyMeshPromises, ...partMeshPromises]);

  // Step 6: Sync initial visibility
  function syncVisibility(): void {
    designScene.syncVisibility();
  }
  syncVisibility();

  // Step 7: Build component tree and wire callbacks
  const tree = new ComponentTree(
    treePanelEl,
    manifest as any,
    (_nodeId: string, _data: unknown) => {
      // Node selected — could focus camera in future
    },
    {
      onToggleNodeHidden: (nodeId: string) => {
        designScene.tree.toggleHidden(nodeId);

        // Lazy-load layer mesh on first show
        const node = designScene.tree.getNode(nodeId);
        if (node && node.kind === NodeKind.Layer && node.meshIds.length === 0 && !node.hidden) {
          lazyLoadLayerMesh(nodeId, manifest, designScene, viewport, botName, bodyGroups).then(() => {
            syncVisibility();
          });
        }

        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onSolo: (nodeId: string) => {
        designScene.tree.solo(nodeId);
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onUnsolo: () => {
        designScene.tree.unsolo();
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
    },
  );
  tree.build();
  tree.updateFromDesignScene(designScene.tree);

  // Step 8: Click-to-select — raycast against body group meshes
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pointerDown = new THREE.Vector2();
  let selectedMesh: THREE.Mesh | null = null;
  const CLICK_THRESHOLD = 5; // pixels — ignore drag gestures

  const canvas = viewport.renderer.domElement;
  canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    pointerDown.set(e.clientX, e.clientY);
  });
  canvas.addEventListener('pointerup', (e: PointerEvent) => {
    // Only treat as click if pointer didn't move much (ignore drags)
    const dx = e.clientX - pointerDown.x;
    const dy = e.clientY - pointerDown.y;
    if (dx * dx + dy * dy > CLICK_THRESHOLD * CLICK_THRESHOLD) return;

    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewport.camera);

    // Collect all visible meshes from body groups
    const targets: THREE.Mesh[] = [];
    for (const group of Object.values(bodyGroups)) {
      group.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && child.visible) {
          targets.push(child as THREE.Mesh);
        }
      });
    }

    const hits = raycaster.intersectObjects(targets, false);

    // Clear previous highlight
    if (selectedMesh) {
      const mat = selectedMesh.material as THREE.MeshPhysicalMaterial;
      if (mat.emissive) mat.emissive.setHex(0x000000);
      selectedMesh = null;
    }

    if (hits.length > 0) {
      const hitMesh = hits[0].object as THREE.Mesh;
      const nodeId = designScene.meshToNode.get(hitMesh.uuid);

      // Highlight mesh with emissive tint
      selectedMesh = hitMesh;
      const mat = hitMesh.material as THREE.MeshPhysicalMaterial;
      if (mat.emissive) mat.emissive.setHex(0x333333);

      // Highlight corresponding tree node
      if (nodeId) {
        tree.setFocused(nodeId);
      }
    } else {
      tree.clearFocus();
    }
  });

  // Step 9: Frame camera on loaded geometry
  const box = new THREE.Box3();
  for (const mesh of designScene.meshes.values()) {
    box.expandByObject(mesh);
  }
  if (!box.isEmpty()) {
    viewport.frameOnBox(box);
  }

  return {
    scene: designScene,
    tree,
    viewport,
    syncVisibility,
  };
}

// ---------------------------------------------------------------------------
// Node ID resolution — maps manifest parts to SceneTree node IDs
// ---------------------------------------------------------------------------

/**
 * Resolve the SceneTree node ID for a manifest part.
 *
 * build-scene-tree.ts uses `component:${part.id}` for servo/mounted parts
 * and `subpart:${part.id}` for horns. Fasteners are grouped under
 * `subpart:` nodes and don't have individual meshes in the tree.
 */
function resolvePartNodeId(part: ManifestPart): string {
  if (part.category === 'horn') {
    return `subpart:${part.id}`;
  }
  if (part.category === 'fastener') {
    // Fasteners are grouped — use the joint-scoped group key
    const groupKey = part.joint ? `subpart:${part.joint}:${part.name}` : `subpart:${part.id}:${part.name}`;
    return groupKey;
  }
  // servo, camera, compute, battery, wheel, component — all use component: prefix
  return `component:${part.id}`;
}
