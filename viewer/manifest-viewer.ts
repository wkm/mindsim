/**
 * ManifestViewer — shared viewing pipeline for bot and component viewers.
 *
 * Given a ViewerManifest and a container, builds the full 3D scene:
 *   manifest → SceneTree → DesignScene → mesh loading → ComponentTree → interaction
 *
 * Both design-viewer.ts and component-browser.ts can delegate to this module
 * instead of duplicating the pipeline.
 */

import * as THREE from 'three';
import { buildSceneTree } from './build-scene-tree.ts';
import { ComponentTree } from './component-tree.ts';
import { DesignScene, NodeKind, type SceneNode } from './design-scene.ts';
import type { ManifestMount, ManifestPart, ViewerManifest } from './manifest-types.ts';
import { fetchSTLFromUrl, makeMaterial, manifestQuatToThree, resolvePartNodeId } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Context for resolving STL URLs — lets callers customize the URL scheme. */
export interface StlUrlContext {
  kind: 'body' | 'mount' | 'mount-material' | 'part';
  name: string;
  componentName: string;
}

export interface ManifestViewerOptions {
  container: HTMLElement;
  treePanelEl: HTMLElement;
  sidePanelEl: HTMLElement;
  manifest: ViewerManifest;
  resolveStlUrl: (mesh: string, context: StlUrlContext) => string;
  onNodeSelected?: (nodeId: string | null, manifest: ViewerManifest) => void;
  onSoloChanged?: (nodeId: string | null) => void;
  viewport?: Viewport3D;
}

export interface ManifestViewerContext {
  scene: DesignScene;
  tree: ComponentTree;
  viewport: Viewport3D;
  syncVisibility(): void;
  dispose(): void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EDGE_LINE_MAT = new THREE.LineBasicMaterial({
  color: 0x000000,
  transparent: true,
  opacity: 0.35,
  linewidth: 2,
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a positioned mesh with edge lines. */
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

  // Only add edge lines for opaque meshes — transparent overlays look better without them
  const matObj = material as THREE.MeshPhysicalMaterial;
  if (!matObj.transparent) {
    const edges = new THREE.EdgesGeometry(geometry, 28);
    const lines = new THREE.LineSegments(edges, EDGE_LINE_MAT);
    lines.raycast = () => {};
    mesh.add(lines);
  }

  return mesh;
}

/** Map a SceneNode to a filter category string. */
function nodeCategoryForFilter(node: SceneNode, manifest: ViewerManifest): string | null {
  if (node.kind === NodeKind.Body) return 'body';
  if (node.kind === NodeKind.SubPart) {
    if (node.id.startsWith('fastener')) return 'fastener';
  }
  if (node.id.startsWith('wire-group:')) return 'wire';
  if (node.kind === NodeKind.Component) {
    const mounts = manifest.mounts ?? [];
    const mount = mounts.find((m: ManifestMount) => `mount:${m.body}:${m.label}` === node.id);
    if (mount?.category === 'design_layer') return 'design_layer';
    if (mount?.category === 'clearance') return 'clearance';
    return 'mount';
  }
  return null;
}

// ---------------------------------------------------------------------------
// Lazy-load helpers
// ---------------------------------------------------------------------------

/** Lazy-load a design layer mesh on first visibility toggle. */
async function lazyLoadLayerMesh(
  nodeId: string,
  manifest: ViewerManifest,
  designScene: DesignScene,
  viewport: Viewport3D,
  resolveStlUrl: ManifestViewerOptions['resolveStlUrl'],
  bodyGroups: Record<string, THREE.Group>,
  layerLoadState: Map<string, 'loading' | 'loaded'>,
): Promise<void> {
  if (layerLoadState.has(nodeId)) return;
  layerLoadState.set(nodeId, 'loading');

  // Parse layer nodeId: "layer:{jointName}:{kind}"
  const parts = nodeId.split(':');
  if (parts.length < 3) return;
  const jointName = parts[1];
  const layerKind = parts[2];

  const joint = manifest.joints.find((j) => j.name === jointName);
  if (!joint?.design_layers) return;

  const layer = joint.design_layers.find((dl) => dl.kind === layerKind);
  if (!layer) return;

  const parentBody = manifest.bodies.find((b) => b.name === layer.parent_body);
  const pos = parentBody?.pos ?? [0, 0, 0];
  const quat = parentBody?.quat ?? [1, 0, 0, 0];

  const layerColors: Record<string, number> = {
    bracket: 0x4488cc,
    coupler: 0x44cc88,
    clearance: 0xcc4444,
    insertion: 0xccaa44,
  };

  const url = resolveStlUrl(layer.mesh, { kind: 'body', name: layer.mesh, componentName: manifest.bot_name });
  const geometry = await fetchSTLFromUrl(url);
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

/** Lazy-load mount or part meshes when a node is first shown. */
async function lazyLoadNodeMesh(
  nodeId: string,
  manifest: ViewerManifest,
  designScene: DesignScene,
  viewport: Viewport3D,
  resolveStlUrl: ManifestViewerOptions['resolveStlUrl'],
  bodyGroups: Record<string, THREE.Group>,
): Promise<void> {
  const allParts = manifest.parts ?? [];

  if (nodeId.startsWith('mount:')) {
    // Design layer or clearance mount — find the matching mount
    const mounts = manifest.mounts ?? [];
    const mount = mounts.find((m: ManifestMount) => `mount:${m.body}:${m.label}` === nodeId);
    if (!mount) return;

    let group = bodyGroups[nodeId];
    if (!group) {
      group = viewport.addGroup(nodeId);
      bodyGroups[nodeId] = group;
    }

    const url = resolveStlUrl(mount.mesh, { kind: 'mount', name: mount.label, componentName: mount.component });
    const geometry = await fetchSTLFromUrl(url);
    if (!geometry) return;

    let color: THREE.Color;
    if (mount.color) {
      color = new THREE.Color(mount.color[0], mount.color[1], mount.color[2]);
    } else {
      color = new THREE.Color(0.5, 0.5, 0.5);
    }

    const isTransparent = mount.color?.[3] !== undefined && mount.color[3] < 1;
    const mat = new THREE.MeshPhysicalMaterial({
      color,
      roughness: 0.5,
      metalness: 0.0,
      transparent: isTransparent,
      opacity: mount.color?.[3] ?? 1.0,
      ...(isTransparent ? { depthWrite: false, side: THREE.DoubleSide } : {}),
    });

    const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
    group.add(mesh);
    designScene.registerMesh(nodeId, mesh);
    return;
  }

  let matchingParts: ManifestPart[];

  if (nodeId.startsWith('wire-group:')) {
    matchingParts = allParts.filter((p) => p.category === 'wire');
  } else if (nodeId.startsWith('fastener-group:')) {
    const segments = nodeId.split(':');
    const groupKey = segments[segments.length - 1];
    matchingParts = allParts.filter((p) => p.category === 'fastener' && p.name === groupKey);
  } else {
    return;
  }

  if (matchingParts.length === 0) return;

  let group = bodyGroups[nodeId];
  if (!group) {
    group = viewport.addGroup(nodeId);
    bodyGroups[nodeId] = group;
  }

  // Deduplicate STL URLs, batch-load geometries
  const urlForPart = (part: ManifestPart) =>
    resolveStlUrl(part.mesh, { kind: 'part', name: part.name, componentName: manifest.bot_name });

  const uniqueUrls = [...new Set(matchingParts.map(urlForPart))];
  const geomCache: Record<string, THREE.BufferGeometry> = {};
  await Promise.all(
    uniqueUrls.map(async (url) => {
      const geom = await fetchSTLFromUrl(url);
      if (geom) geomCache[url] = geom;
    }),
  );

  for (const part of matchingParts) {
    const url = urlForPart(part);
    const srcGeom = geomCache[url];
    if (!srcGeom) continue;

    let mat: THREE.Material;
    if (part.category === 'fastener') {
      mat = new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.2 });
    } else if (part.color) {
      const c = new THREE.Color(part.color[0], part.color[1], part.color[2]);
      mat = new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5, emissive: c, emissiveIntensity: 0.15 });
    } else {
      mat = new THREE.MeshPhysicalMaterial({ color: 0x808080, roughness: 0.5, metalness: 0.0 });
    }

    const mesh = createPositionedMesh(srcGeom.clone(), mat, part.pos ?? [0, 0, 0], part.quat ?? [1, 0, 0, 0]);
    group.add(mesh);
    designScene.registerMesh(nodeId, mesh);
  }
}

// ---------------------------------------------------------------------------
// Section plane
// ---------------------------------------------------------------------------

interface SectionState {
  enabled: boolean;
  axis: string;
  flipped: boolean;
  fraction: number;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export async function initManifestViewer(options: ManifestViewerOptions): Promise<ManifestViewerContext> {
  const {
    container,
    treePanelEl,
    sidePanelEl: _sidePanelEl,
    manifest,
    resolveStlUrl,
    onNodeSelected,
    onSoloChanged,
  } = options;
  const materials = manifest.materials ?? {};

  // Layer lazy-loading state
  const layerLoadState = new Map<string, 'loading' | 'loaded'>();

  // Section plane UI state
  const section: SectionState = {
    enabled: false,
    axis: 'z',
    flipped: false,
    fraction: 0.5,
  };

  // Track DOM listeners for cleanup in dispose()
  const domListenerCleanups: (() => void)[] = [];
  function addDomListener(el: EventTarget, event: string, handler: EventListener): void {
    el.addEventListener(event, handler);
    domListenerCleanups.push(() => el.removeEventListener(event, handler));
  }

  // -----------------------------------------------------------------------
  // Step 1: Viewport — reuse or create
  // -----------------------------------------------------------------------
  const viewport = options.viewport ?? new Viewport3D(container, { cameraType: 'orthographic', grid: true });

  // -----------------------------------------------------------------------
  // Step 2: SceneTree + DesignScene
  // -----------------------------------------------------------------------
  const sceneTree = buildSceneTree(manifest);
  const designScene = new DesignScene(sceneTree);

  // -----------------------------------------------------------------------
  // Step 3: One group per body
  // -----------------------------------------------------------------------
  const bodyGroups: Record<string, THREE.Group> = {};
  for (const body of manifest.bodies) {
    bodyGroups[body.name] = viewport.addGroup(`body:${body.name}`);
  }

  function getBodyGroup(bodyName: string): THREE.Group {
    return bodyGroups[bodyName] ?? Object.values(bodyGroups)[0];
  }

  // -----------------------------------------------------------------------
  // Step 4: Load body meshes
  // -----------------------------------------------------------------------
  const bodyMeshPromises = manifest.bodies.map(async (body) => {
    const url = resolveStlUrl(body.mesh, { kind: 'body', name: body.name, componentName: manifest.bot_name });
    const geometry = await fetchSTLFromUrl(url);
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
    const nodeId = `body:${body.name}`;
    getBodyGroup(body.name).add(mesh);
    designScene.registerMesh(nodeId, mesh);
  });

  // -----------------------------------------------------------------------
  // Step 5a: Load mount meshes
  // -----------------------------------------------------------------------
  const mountsList = manifest.mounts ?? [];
  const mountMeshPromises = mountsList.map(async (mount) => {
    if (!mount.pos || !mount.quat) return;

    const nodeId = `mount:${mount.body}:${mount.label}`;
    const group = getBodyGroup(mount.body);

    if (mount.meshes && mount.meshes.length > 0) {
      const subPromises = mount.meshes.map(async (sub) => {
        const url = resolveStlUrl(sub.file, {
          kind: 'mount-material',
          name: sub.material,
          componentName: mount.component,
        });
        const geometry = await fetchSTLFromUrl(url);
        if (!geometry) return;

        const matDef = materials[sub.material];
        const mat = makeMaterial(matDef);
        const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
        group.add(mesh);
        designScene.registerMesh(nodeId, mesh);
      });
      await Promise.all(subPromises);
    } else {
      const url = resolveStlUrl(mount.mesh, { kind: 'mount', name: mount.label, componentName: mount.component });
      const geometry = await fetchSTLFromUrl(url);
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
      designScene.registerMesh(nodeId, mesh);
    }
  });

  // -----------------------------------------------------------------------
  // Step 5b: Load part meshes (fasteners, wires)
  // -----------------------------------------------------------------------
  const partsList = manifest.parts ?? [];
  const partMeshPromises = partsList.map(async (part) => {
    if (!part.pos || !part.quat) return;

    const nodeId = resolvePartNodeId(part);
    const group = getBodyGroup(part.parent_body);

    const url = resolveStlUrl(part.mesh, { kind: 'part', name: part.name, componentName: manifest.bot_name });
    const geometry = await fetchSTLFromUrl(url);
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
    designScene.registerMesh(nodeId, mesh);
  });

  // -----------------------------------------------------------------------
  // Step 6: Wait for all eager meshes
  // -----------------------------------------------------------------------
  await Promise.all([...bodyMeshPromises, ...mountMeshPromises, ...partMeshPromises]);

  // -----------------------------------------------------------------------
  // Step 7: Sync visibility helper
  // -----------------------------------------------------------------------
  function syncVisibility(): void {
    designScene.syncVisibility();
  }
  syncVisibility();

  // -----------------------------------------------------------------------
  // Step 8: Build component tree
  // -----------------------------------------------------------------------
  const tree = new ComponentTree(
    treePanelEl,
    manifest,
    (nodeId: string, _data: unknown) => {
      if (onNodeSelected) onNodeSelected(nodeId, manifest);
    },
    {
      onToggleNodeHidden: (nodeId: string) => {
        designScene.tree.toggleHidden(nodeId);

        const node = designScene.tree.getNode(nodeId);
        if (node && !node.hidden && node.meshIds.length === 0) {
          if (node.kind === NodeKind.Layer) {
            // Design layer (bot joint layers) — separate lazy-load path
            lazyLoadLayerMesh(nodeId, manifest, designScene, viewport, resolveStlUrl, bodyGroups, layerLoadState).then(
              () => syncVisibility(),
            );
          } else {
            // Mount/part meshes — generic lazy-load
            lazyLoadNodeMesh(nodeId, manifest, designScene, viewport, resolveStlUrl, bodyGroups).then(() =>
              syncVisibility(),
            );
          }
        }

        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onCategoryToggle: (category: string, visible: boolean) => {
        for (const node of designScene.tree.nodes.values()) {
          const nodeCategory = nodeCategoryForFilter(node, manifest);
          if (nodeCategory === category) {
            node.hidden = !visible;
          }
        }
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onSolo: (nodeId: string) => {
        if (designScene.tree.soloedId === nodeId) {
          designScene.tree.unsolo();
          if (onSoloChanged) onSoloChanged(null);
        } else {
          designScene.tree.solo(nodeId);
          if (onSoloChanged) onSoloChanged(nodeId);
        }
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onUnsolo: () => {
        designScene.tree.unsolo();
        if (onSoloChanged) onSoloChanged(null);
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
    },
  );
  tree.build();
  tree.updateFromDesignScene(designScene.tree);

  // -----------------------------------------------------------------------
  // Step 9: Default visibility — hide fasteners, wires, design layers, clearances
  // -----------------------------------------------------------------------
  const hiddenMountIds = new Set<string>();
  for (const m of manifest.mounts ?? []) {
    if (m.category === 'design_layer' || m.category === 'clearance') {
      hiddenMountIds.add(`mount:${m.body}:${m.label}`);
    }
  }
  for (const node of designScene.tree.nodes.values()) {
    if (node.id.startsWith('fastener-group:') || node.id.startsWith('wire-group:') || hiddenMountIds.has(node.id)) {
      node.hidden = true;
    }
  }
  syncVisibility();
  tree.updateFromDesignScene(designScene.tree);

  // -----------------------------------------------------------------------
  // Step 10: Section cap colors
  // -----------------------------------------------------------------------
  viewport.setSectionCapColorFn((groupName: string) => {
    if (groupName.startsWith('body:')) {
      const body = manifest.bodies.find((b) => `body:${b.name}` === groupName);
      if (body?.color) {
        return new THREE.Color(body.color[0], body.color[1], body.color[2]);
      }
      return 0xced9e0;
    }
    if (groupName.startsWith('fastener')) return 0xd4a843;
    if (groupName.startsWith('wire') || groupName.startsWith('connector')) return 0x9179f2;

    // Mount color
    const mounts = manifest.mounts ?? [];
    for (const m of mounts) {
      if (`mount:${m.body}:${m.label}` === groupName && m.color) {
        return new THREE.Color(m.color[0], m.color[1], m.color[2]);
      }
    }

    return 0xced9e0;
  });

  // -----------------------------------------------------------------------
  // Step 11: Section plane UI
  // -----------------------------------------------------------------------
  const sectionToggle = document.getElementById('section-toggle');
  if (sectionToggle) {
    addDomListener(sectionToggle, 'click', () => {
      section.enabled = !section.enabled;
      sectionToggle.classList.toggle('active', section.enabled);
      const controls = document.getElementById('section-controls');
      if (controls) controls.style.display = section.enabled ? 'flex' : 'none';
      viewport.setSection(section);
    });
  }

  for (const axis of ['x', 'y', 'z']) {
    const btn = document.querySelector(`[data-section-axis="${axis}"]`);
    if (btn) {
      addDomListener(btn, 'click', () => {
        section.axis = axis;
        document.querySelectorAll('[data-section-axis]').forEach((b) => {
          b.classList.toggle('active', (b as HTMLElement).dataset.sectionAxis === axis);
        });
        viewport.setSection(section);
      });
    }
  }

  const slider = document.getElementById('section-slider');
  if (slider) {
    addDomListener(slider, 'input', () => {
      section.fraction = Number.parseFloat((slider as HTMLInputElement).value) / 100;
      viewport.setSection(section);
    });
  }

  const flipBtn = document.getElementById('section-flip');
  if (flipBtn) {
    addDomListener(flipBtn, 'click', () => {
      section.flipped = !section.flipped;
      flipBtn.classList.toggle('active', section.flipped);
      viewport.setSection(section);
    });
  }

  // -----------------------------------------------------------------------
  // Step 12: Measure tool UI
  // -----------------------------------------------------------------------
  const measureBtn = document.getElementById('measure-toggle');
  if (measureBtn) {
    addDomListener(measureBtn, 'click', () => {
      const active = !viewport.measureEnabled;
      if (active) {
        viewport.enableMeasureTool();
      } else {
        viewport.disableMeasureTool();
      }
      measureBtn.classList.toggle('active', active);
      const clearEl = document.getElementById('measure-clear');
      if (clearEl) clearEl.style.display = active ? '' : 'none';
    });
  }

  const clearBtn = document.getElementById('measure-clear');
  if (clearBtn) {
    addDomListener(clearBtn, 'click', () => {
      viewport.clearMeasurements();
    });
  }

  // -----------------------------------------------------------------------
  // Step 13: Keyboard shortcuts (S = section, M = measure)
  // -----------------------------------------------------------------------
  const onKeyDown = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return;
    const ctrl = e.ctrlKey || e.metaKey;

    if (e.key === 's' && !ctrl) {
      e.preventDefault();
      sectionToggle?.click();
    } else if (e.key === 'm' && !ctrl) {
      e.preventDefault();
      measureBtn?.click();
    }
  };
  document.addEventListener('keydown', onKeyDown);

  // -----------------------------------------------------------------------
  // Step 14: Click-to-select
  // -----------------------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pointerDown = new THREE.Vector2();
  let selectedMesh: THREE.Mesh | null = null;
  const CLICK_THRESHOLD = 5;

  const canvas = viewport.renderer.domElement;

  const onPointerDown = (e: PointerEvent) => {
    pointerDown.set(e.clientX, e.clientY);
  };

  const onPointerUp = (e: PointerEvent) => {
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

      selectedMesh = hitMesh;
      const mat = hitMesh.material as THREE.MeshPhysicalMaterial;
      if (mat.emissive) mat.emissive.setHex(0x333333);

      if (nodeId) {
        tree.setFocused(nodeId);
        if (onNodeSelected) onNodeSelected(nodeId, manifest);
      }
    } else {
      tree.clearFocus();
      if (onNodeSelected) onNodeSelected(null, manifest);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);

  // -----------------------------------------------------------------------
  // Step 15: Frame camera on loaded geometry
  // -----------------------------------------------------------------------
  const box = new THREE.Box3();
  for (const mesh of designScene.meshes.values()) {
    box.expandByObject(mesh);
  }
  if (!box.isEmpty()) {
    viewport.frameOnBox(box);
  }

  // -----------------------------------------------------------------------
  // Step 16: Start animation
  // -----------------------------------------------------------------------
  viewport.animate(() => {});

  // -----------------------------------------------------------------------
  // Step 17: Dispose
  // -----------------------------------------------------------------------
  function dispose(): void {
    // Remove pointer + keyboard listeners
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerup', onPointerUp);
    document.removeEventListener('keydown', onKeyDown);
    // Remove section/measure DOM listeners
    for (const cleanup of domListenerCleanups) cleanup();
    domListenerCleanups.length = 0;
    // Dispose tree
    tree.dispose();
    // Clean up mesh groups via viewport (keeps viewport's internal registry clean)
    for (const group of Object.values(bodyGroups)) {
      viewport.clearGroup(group);
      viewport.scene.remove(group);
    }
  }

  return {
    scene: designScene,
    tree,
    viewport,
    syncVisibility,
    dispose,
  };
}
