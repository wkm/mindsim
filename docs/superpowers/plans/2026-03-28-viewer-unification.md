# Viewer Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a shared `ManifestViewer` from the duplicated bot/component viewer pipelines, making both viewers thin wrappers around it.

**Architecture:** `manifest-viewer.ts` owns the full viewing experience (manifest → SceneTree → DesignScene → mesh loading → ComponentTree → click-to-select → section plane → measure tool). `design-viewer.ts` and `component-browser.ts` become thin callers that fetch their manifest, provide a `resolveStlUrl` callback, and handle context-specific UI (sim tab, specs panel, steps mode).

**Tech Stack:** TypeScript, Three.js, existing Viewport3D/ComponentTree/DesignScene infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `viewer/manifest-viewer.ts` | **Create** | Shared viewer: manifest→scene→meshes→tree→tools→raycasting |
| `viewer/design-viewer.ts` | **Modify** | Thin bot wrapper: fetch manifest, provide resolveStlUrl, onNodeSelected, sim tab |
| `viewer/component-browser.ts` | **Modify** | Thin component wrapper: fetch manifest, provide resolveStlUrl, specs panel, markers, steps mode |
| `viewer/viewer.ts` | **Modify** | Converge layout: both paths use tree-panel + side-panel + canvas |
| `viewer/utils.ts` | **Modify** | Generalize `fetchSTL` to accept a URL directly (not hardcode bot path) |

---

## Task 1: Create `manifest-viewer.ts` — core viewing pipeline

**Files:**
- Create: `viewer/manifest-viewer.ts`
- Modify: `viewer/utils.ts` (add `fetchSTLFromUrl` helper)

This is the heart of the refactor. Extract the shared pipeline from `design-viewer.ts` into a new module.

- [ ] **Step 1: Add `fetchSTLFromUrl` to utils.ts**

The existing `fetchSTL(botName, meshFile)` hardcodes `/api/bots/{bot}/meshes/{meshFile}`. Add a general-purpose version that takes a full URL:

```typescript
// In viewer/utils.ts, next to the existing fetchSTL
export async function fetchSTLFromUrl(url: string): Promise<THREE.BufferGeometry | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      console.warn(`[viewer] failed to fetch STL: ${url} (${resp.status})`);
      return null;
    }
    const buf = await resp.arrayBuffer();
    const geometry = stlLoader.parse(buf);
    geometry.computeVertexNormals();
    return geometry;
  } catch (err) {
    console.warn(`[viewer] STL fetch error: ${url}`, err);
    return null;
  }
}
```

- [ ] **Step 2: Create `manifest-viewer.ts` with types and skeleton**

```typescript
// viewer/manifest-viewer.ts
import * as THREE from 'three';
import { buildSceneTree } from './build-scene-tree.ts';
import { ComponentTree } from './component-tree.ts';
import { DesignScene, NodeKind, type SceneNode } from './design-scene.ts';
import type { ViewerManifest, ManifestMount, ManifestPart, ManifestBody } from './manifest-types.ts';
import { addMeshWithEdges, createMaterial, tintColor } from './presentation.ts';
import { fetchSTLFromUrl } from './utils.ts';
import { Viewport3D } from './viewport3d.ts';

export interface StlUrlContext {
  kind: 'body' | 'mount' | 'mount-material' | 'part';
  name: string;
  componentName: string;
}

export interface ManifestViewerOptions {
  container: HTMLElement;       // Canvas container element
  treePanelEl: HTMLElement;     // Tree panel DOM element (left sidebar)
  sidePanelEl: HTMLElement;     // Side panel DOM element (right sidebar)
  manifest: ViewerManifest;
  resolveStlUrl: (mesh: string, context: StlUrlContext) => string;
  onNodeSelected?: (nodeId: string | null, manifest: ViewerManifest) => void;
  viewport?: Viewport3D;        // Optional — if provided, reuse instead of creating new
}

export interface ManifestViewerContext {
  scene: DesignScene;
  tree: ComponentTree;
  viewport: Viewport3D;
  syncVisibility(): void;
  dispose(): void;
}

export async function initManifestViewer(options: ManifestViewerOptions): Promise<ManifestViewerContext> {
  // Implementation in next steps
}
```

- [ ] **Step 3: Implement Viewport3D creation and SceneTree building**

Inside `initManifestViewer`:

```typescript
  const { container, treePanelEl, sidePanelEl, manifest, resolveStlUrl, onNodeSelected } = options;
  const materials = manifest.materials ?? {};

  // Reuse provided viewport or create a new one (orthographic by default)
  const viewport = options.viewport ?? new Viewport3D(container, { cameraType: 'orthographic', grid: true });

  // Build scene tree + design scene
  const sceneTree = buildSceneTree(manifest);
  const designScene = new DesignScene(sceneTree);

  // Create a viewport group per body
  const meshGroups: Record<string, THREE.Group> = {};
  for (const body of manifest.bodies) {
    meshGroups[`body:${body.name}`] = viewport.addGroup(`body:${body.name}`);
  }

  function getBodyGroup(bodyName: string): THREE.Group {
    return meshGroups[`body:${bodyName}`] ?? Object.values(meshGroups)[0];
  }
```

- [ ] **Step 4: Implement mesh loading (bodies, mounts, parts)**

Add mesh loading that uses `resolveStlUrl` instead of hardcoded URLs. Follow the `design-viewer.ts` pattern but use the callback for URL resolution.

```typescript
  // Helper: create positioned mesh with edge lines
  const EDGE_LINE_MAT = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 });
  function createPositionedMesh(
    geometry: THREE.BufferGeometry, material: THREE.Material,
    pos: number[], quat: number[],
  ): THREE.Mesh {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set(pos[0], pos[1], pos[2]);
    // Manifest quat is wxyz, Three.js is xyzw
    mesh.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
    const edges = new THREE.EdgesGeometry(geometry, 28);
    const lines = new THREE.LineSegments(edges, EDGE_LINE_MAT);
    lines.raycast = () => {};
    mesh.add(lines);
    return mesh;
  }

  const compName = manifest.bot_name;

  // Load body meshes
  const bodyPromises = manifest.bodies.map(async (body) => {
    const url = resolveStlUrl(body.mesh, { kind: 'body', name: body.name, componentName: compName });
    const geometry = await fetchSTLFromUrl(url);
    if (!geometry) return;

    const bodyColor = body.color
      ? new THREE.Color(body.color[0], body.color[1], body.color[2])
      : new THREE.Color(body.role === 'component' ? 0x808080 : 0xd9d9d9);
    const mat = new THREE.MeshPhysicalMaterial({
      color: bodyColor, roughness: 0.6, metalness: 0.0,
    });
    const mesh = createPositionedMesh(geometry, mat, body.pos, body.quat);
    getBodyGroup(body.name).add(mesh);
    designScene.registerMesh(`body:${body.name}`, mesh);
  });

  // Load mount meshes (multi-material or single)
  const mountsList = manifest.mounts ?? [];
  const mountPromises = mountsList.map(async (mount) => {
    const nodeId = `mount:${mount.body}:${mount.label}`;
    const group = getBodyGroup(mount.body);

    if (mount.meshes && mount.meshes.length > 0) {
      const subPromises = mount.meshes.map(async (sub) => {
        const url = resolveStlUrl(sub.file, { kind: 'mount-material', name: mount.label, componentName: compName });
        const geometry = await fetchSTLFromUrl(url);
        if (!geometry) return;
        const matDef = materials[sub.material];
        const color = matDef ? new THREE.Color(matDef.color[0], matDef.color[1], matDef.color[2]) : new THREE.Color(0x808080);
        const mat = new THREE.MeshPhysicalMaterial({
          color, roughness: matDef?.roughness ?? 0.5, metalness: matDef?.metallic ?? 0,
        });
        const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
        group.add(mesh);
        designScene.registerMesh(nodeId, mesh);
      });
      await Promise.all(subPromises);
    } else {
      const url = resolveStlUrl(mount.mesh, { kind: 'mount', name: mount.label, componentName: compName });
      const geometry = await fetchSTLFromUrl(url);
      if (!geometry) return;
      const mountColor = mount.color
        ? new THREE.Color(mount.color[0], mount.color[1], mount.color[2])
        : new THREE.Color(0x808080);
      const opts: any = {};
      if (mount.color && mount.color[3] !== undefined && mount.color[3] < 1) {
        opts.transparent = true;
        opts.opacity = mount.color[3];
      }
      const mat = new THREE.MeshPhysicalMaterial({
        color: mountColor, roughness: 0.5, metalness: 0, ...opts,
      });
      const mesh = createPositionedMesh(geometry, mat, mount.pos, mount.quat);
      group.add(mesh);
      designScene.registerMesh(nodeId, mesh);
    }
  });

  // Load part meshes (fasteners, wires)
  const partsList = manifest.parts ?? [];
  const partPromises = partsList.map(async (part) => {
    if (!part.pos || !part.quat) return;
    const nodeId = resolvePartNodeId(part);
    const group = getBodyGroup(part.parent_body);
    const url = resolveStlUrl(part.mesh, { kind: 'part', name: part.id, componentName: compName });
    const geometry = await fetchSTLFromUrl(url);
    if (!geometry) return;

    let mat: THREE.Material;
    if (part.category === 'fastener') {
      mat = new THREE.MeshPhysicalMaterial({ color: 0xc0c0c0, metalness: 0.9, roughness: 0.2 });
    } else if (part.color) {
      const c = new THREE.Color(part.color[0], part.color[1], part.color[2]);
      mat = new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.5 });
    } else {
      mat = new THREE.MeshPhysicalMaterial({ color: 0x808080, roughness: 0.5 });
    }
    const mesh = createPositionedMesh(geometry, mat, part.pos, part.quat);
    group.add(mesh);
    designScene.registerMesh(nodeId, mesh);
  });

  await Promise.all([...bodyPromises, ...mountPromises, ...partPromises]);
```

**Before this step:** Move `resolvePartNodeId` from `design-viewer.ts` (line ~398, currently module-private) to `viewer/utils.ts` with an `export`. This function maps a ManifestPart to its tree node ID (e.g., `fastener-group:mount:{body}:{label}:{name}` or `wire-group:{body}`). Update `design-viewer.ts` to import it from utils.ts. Both `manifest-viewer.ts` and `design-viewer.ts` will import from utils.ts.

- [ ] **Step 5: Implement ComponentTree wiring + default visibility**

```typescript
  // Sync helper
  function syncVisibility(): void {
    designScene.syncVisibility();
  }

  // Build ComponentTree
  const tree = new ComponentTree(treePanelEl, manifest,
    (nodeId: string) => { onNodeSelected?.(nodeId, manifest); },
    {
      onToggleNodeHidden: (nodeId: string) => {
        designScene.tree.toggleHidden(nodeId);

        // Lazy-load mount/layer mesh on first show
        const node = designScene.tree.getNode(nodeId);
        if (node && node.meshIds.length === 0 && !node.hidden) {
          lazyLoadMesh(nodeId).then(() => syncVisibility());
        }

        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onCategoryToggle: (category: string, visible: boolean) => {
        for (const node of designScene.tree.nodes.values()) {
          if (matchesCategory(node, category, manifest)) {
            node.hidden = !visible;
          }
        }
        syncVisibility();
        tree.updateFromDesignScene(designScene.tree);
      },
      onSolo: (nodeId: string) => {
        if (designScene.tree.soloedId === nodeId) designScene.tree.unsolo();
        else designScene.tree.solo(nodeId);
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

  // Default visibility: hide fastener groups, wire groups, design layers, clearances
  const hiddenMountIds = new Set<string>();
  for (const m of mountsList) {
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
```

- [ ] **Step 6: Implement section plane, measure tool, section cap colors**

Move section plane state + update logic from component-browser.ts into ManifestViewer. The section toolbar HTML buttons already exist in index.html — ManifestViewer wires them up.

```typescript
  // Section plane state
  let sectionEnabled = false;
  let sectionAxis = 'z';
  let sectionFlipped = false;
  let sectionFraction = 0.5;

  function getVisibleBBox(): THREE.Box3 | null {
    const box = new THREE.Box3();
    let has = false;
    for (const group of Object.values(meshGroups)) {
      if (!group.visible) continue;
      group.traverse((child) => {
        const m = child as THREE.Mesh;
        if (m.isMesh) {
          m.geometry.computeBoundingBox();
          const cb = m.geometry.boundingBox!.clone();
          cb.applyMatrix4(m.matrixWorld);
          box.union(cb);
          has = true;
        }
      });
    }
    return has ? box : null;
  }

  function updateSectionPlane() {
    // Copy the full implementation from component-browser.ts _updateSectionPlane
    // (lines 1436-1483). It computes the clipping plane from sectionAxis,
    // sectionFraction, sectionFlipped and the visible bounding box, then sets
    // viewport._secOn, _secPlane.normal, _secPlane.constant, and calls
    // viewport._applySection(). When disabling, it calls viewport._clearSectionCaps()
    // and clears clippingPlanes on all materials.
    //
    // The implementer MUST read component-browser.ts:1436-1483 and copy the
    // full function body — this is ~45 lines of section plane math + viewport
    // state management. Do NOT stub it.
  }

  // Section cap colors derived from manifest
  viewport.setSectionCapColorFn((groupName: string) => {
    // Find the body/mount color from manifest
    for (const body of manifest.bodies) {
      if (groupName === `body:${body.name}`) {
        const c = body.color ?? [0.808, 0.851, 0.878];
        return tintColor(new THREE.Color(c[0], c[1], c[2]), 0.6);
      }
    }
    // Mount groups (fasteners, wires, design layers) — match by group name prefix
    if (groupName.startsWith('fastener')) return tintColor(0xd4a843, 0.6);
    if (groupName.startsWith('wire') || groupName.startsWith('connector')) return tintColor(0x9179f2, 0.6);
    return tintColor(0xced9e0, 0.6);
  });

  // Wire section toolbar buttons (S-key, axis buttons, slider, flip button).
  // Copy from component-browser.ts _setupViewToolbar (lines 225-265).
  // The buttons are: #section-toggle (S-key), [data-section-axis="x/y/z"],
  // #section-slider (input range), #section-flip.
  // Each calls updateSectionPlane() after updating state.

  // Wire measure tool buttons (M-key).
  // Copy from component-browser.ts _setupViewToolbar (lines 188-208).
  // The buttons are: #measure-toggle (M-key), #measure-clear.
  // Toggle calls viewport.enableMeasureTool() / disableMeasureTool().

  // Wire keyboard shortcuts: S for section, M for measure.
  // Copy from component-browser.ts keyboard handler (lines 277-289).
  // Store the keydown listener reference for dispose() cleanup.
```

- [ ] **Step 7: Implement click-to-select raycasting**

```typescript
  let selectedMesh: THREE.Mesh | null = null;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const pointerDown = new THREE.Vector2();
  const CLICK_THRESHOLD = 5;
  const canvas = viewport.renderer.domElement;

  const onPointerDown = (e: PointerEvent) => pointerDown.set(e.clientX, e.clientY);
  const onPointerUp = (e: PointerEvent) => {
    const dx = e.clientX - pointerDown.x;
    const dy = e.clientY - pointerDown.y;
    if (dx * dx + dy * dy > CLICK_THRESHOLD * CLICK_THRESHOLD) return;

    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, viewport.camera);

    const targets: THREE.Mesh[] = [];
    for (const group of Object.values(meshGroups)) {
      group.traverse((child) => {
        if ((child as THREE.Mesh).isMesh && child.visible) targets.push(child as THREE.Mesh);
      });
    }

    const hits = raycaster.intersectObjects(targets, false);

    if (selectedMesh) {
      const mat = selectedMesh.material as THREE.MeshPhysicalMaterial;
      if (mat.emissive) mat.emissive.setHex(0x000000);
      selectedMesh = null;
    }

    if (hits.length > 0) {
      const hitMesh = hits[0].object as THREE.Mesh;
      selectedMesh = hitMesh;
      const mat = hitMesh.material as THREE.MeshPhysicalMaterial;
      if (mat.emissive) mat.emissive.setHex(0x333333);
      const nodeId = designScene.meshToNode.get(hitMesh.uuid);
      if (nodeId) tree.setFocused(nodeId);
      onNodeSelected?.(nodeId ?? null, manifest);
    } else {
      tree.clearFocus();
      onNodeSelected?.(null, manifest);
    }
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
```

- [ ] **Step 8: Implement lazy-load for mount/layer meshes + category matching**

```typescript
  // Lazy-load mesh for nodes not eagerly loaded (design layers, etc.)
  async function lazyLoadMesh(nodeId: string): Promise<void> {
    if (nodeId.startsWith('wire-group:')) {
      const parts = partsList.filter((p) => p.category === 'wire');
      await loadPartsIntoGroup(nodeId, parts);
    } else if (nodeId.startsWith('fastener-group:')) {
      const segments = nodeId.split(':');
      const groupKey = segments[segments.length - 1];
      const parts = partsList.filter((p) => p.category === 'fastener' && p.name === groupKey);
      await loadPartsIntoGroup(nodeId, parts);
    } else if (nodeId.startsWith('mount:')) {
      const mount = mountsList.find((m) => `mount:${m.body}:${m.label}` === nodeId);
      if (mount) await loadMountMesh(nodeId, mount);
    } else if (nodeId.startsWith('layer:')) {
      // Design layer (bot viewer joint layers).
      // Copy the lazyLoadLayerMesh logic from design-viewer.ts (lines 57-120).
      // It parses "layer:{jointName}:{kind}", finds the matching design_layer
      // in manifest.joints[].design_layers, fetches the STL via resolveStlUrl,
      // creates a semi-transparent material, and registers with designScene.
      // The function needs: manifest, designScene, resolveStlUrl, meshGroups.
      // All of these are in scope within initManifestViewer.
      await loadLayerMesh(nodeId);
    }
  }

  // Category matching for filter chips
  function matchesCategory(node: SceneNode, category: string, manifest: ViewerManifest): boolean {
    if (node.id.startsWith('fastener-group:') && category === 'fastener') return true;
    if (node.id.startsWith('wire-group:') && category === 'wire') return true;
    if (category === 'design_layer' || category === 'clearance') {
      const mount = mountsList.find((m) => `mount:${m.body}:${m.label}` === node.id);
      return mount?.category === category;
    }
    return false;
  }
```

- [ ] **Step 9: Implement dispose + frame camera + return context**

```typescript
  // Frame camera
  viewport.animate(() => {});
  const box = new THREE.Box3();
  for (const mesh of designScene.meshes.values()) box.expandByObject(mesh);
  if (!box.isEmpty()) viewport.frameOnBox(box);

  // Dispose function
  function dispose(): void {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointerup', onPointerUp);
    // Remove keyboard listeners (section, measure)
    // ... (store references and remove them)
    tree.dispose();
    for (const group of Object.values(meshGroups)) {
      viewport.clearGroup(group);
    }
  }

  return { scene: designScene, tree, viewport, meshGroups, syncVisibility, dispose };
```

- [ ] **Step 10: Run lint**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`

- [ ] **Step 11: Commit**

```bash
git add viewer/manifest-viewer.ts viewer/utils.ts
git commit -m "feat: create ManifestViewer — shared viewing pipeline

Extracts manifest→SceneTree→DesignScene→mesh loading→ComponentTree→
click-to-select→section plane→measure tool into a reusable module.
Both bot and component viewers will use this as their core."
```

---

## Task 2: Migrate `design-viewer.ts` to use ManifestViewer

**Files:**
- Modify: `viewer/design-viewer.ts`

- [ ] **Step 1: Replace the body of `initDesignViewer` with ManifestViewer**

Keep the function signature. Fetch the manifest, then delegate to `initManifestViewer`:

```typescript
export async function initDesignViewer(
  botName: string,
  viewport: Viewport3D,
  treePanelEl: HTMLElement,
): Promise<DesignViewerContext> {
  const resp = await fetch(`/api/bots/${botName}/viewer_manifest`);
  if (!resp.ok) throw new Error(`Failed to fetch viewer manifest: ${resp.status}`);
  const manifest: ViewerManifest = await resp.json();

  // Pass the existing container and viewport from viewer.ts
  const container = document.getElementById('canvas-container')!;
  const sidePanelEl = document.getElementById('side-panel')!;

  const ctx = await initManifestViewer({
    container,
    treePanelEl,
    sidePanelEl,
    manifest,
    viewport,  // Reuse the viewport created by viewer.ts
    resolveStlUrl: (mesh) => `/api/bots/${botName}/meshes/${mesh}`,
    onNodeSelected: (nodeId) => {
      // Bot-specific selection handling (future: show component specs)
    },
  });

  return {
    scene: ctx.scene,
    tree: ctx.tree,
    viewport: ctx.viewport,
    syncVisibility: ctx.syncVisibility,
  };
}

- [ ] **Step 2: Remove duplicated code from design-viewer.ts**

Delete all the mesh loading, raycasting, tree callback wiring, and syncVisibility code that now lives in ManifestViewer. Keep:
- The `initDesignViewer` wrapper function
- The `DesignViewerContext` interface (maps to ManifestViewerContext)
- The `lazyLoadLayerMesh` function if ManifestViewer doesn't fully handle bot-style layers yet
- Helper types/functions used only by bot context

- [ ] **Step 3: Run lint + verify bot viewer works**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`
Manual: Open `?bot=wheeler_arm`, verify tree, click-select, visibility toggles work.

- [ ] **Step 4: Commit**

```bash
git add viewer/design-viewer.ts viewer/manifest-viewer.ts
git commit -m "refactor: design-viewer delegates to ManifestViewer

initDesignViewer now fetches manifest and calls initManifestViewer.
Removes ~250 lines of duplicated mesh loading, raycasting, and
tree callback code."
```

---

## Task 3: Migrate `component-browser.ts` to use ManifestViewer

**Files:**
- Modify: `viewer/component-browser.ts`
- Modify: `viewer/viewer.ts`

- [ ] **Step 1: Update `viewer.ts` layout for component path**

The component path should use the same tree-panel + side-panel layout as the bot path. Change the `?component` branch to:
- Show `#tree-panel` (left) — ManifestViewer populates it
- Show `#side-panel` (right) — component context populates it
- Hide `#component-browser` wrapper (or remove references)
- Offset `#canvas-container` left by tree panel width (280px)

- [ ] **Step 2: Slim down `component-browser.ts`**

Replace `loadComponent` to:
1. Fetch manifest from `/api/components/${name}/manifest`
2. Call `initManifestViewer` with component-specific `resolveStlUrl`
3. Provide `onNodeSelected` that populates the specs side panel
4. After ManifestViewer init, hide the ghost body node in tree
5. Keep steps mode as a separate toggle (swap ManifestViewer scene with steps groups)
6. Keep markers + specs panel as component context

Remove from component-browser.ts:
- `_setupClickToSelect` (now in ManifestViewer)
- `_onTreeToggle`, `_onTreeSolo`, `_onTreeUnsolo`, `_onCategoryToggle`, `_nodeCategoryForFilter` (now in ManifestViewer)
- `_lazyLoadNodeMesh`, `_loadMultiMaterialMount` (now in ManifestViewer)
- Section plane code: `_updateSectionPlane`, `_resetSection`, section state fields (now in ManifestViewer)
- Measure tool setup (now in ManifestViewer)
- `_getVisibleBBox` (now in ManifestViewer)
- `_meshGroups`, `_designScene`, `_componentTree` fields (owned by ManifestViewer)

Keep in component-browser.ts:
- `_updateSidePanel` (specs panel)
- `_createMarkers`, `_highlightMarker` (3D markers)
- Steps mode (`_toggleStepsMode`, `_enterStepsMode`, `_exitStepsMode`, `_showComponentStep`, etc.)
- `_fetchCatalog`, component catalog management
- Axis gizmo (`_setupAxisGizmo`, `_updateAxisGizmo`)
- Quick switcher integration
- View toolbar setup for component-specific buttons (steps, render SVG)

- [ ] **Step 3: Run lint + verify component viewer works**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`
Manual: Open `?component=STS3215`, verify tree in left panel, specs in right panel, section, measure, click-select, steps mode, markers.

- [ ] **Step 4: Commit**

```bash
git add viewer/component-browser.ts viewer/viewer.ts viewer/manifest-viewer.ts
git commit -m "refactor: component-browser delegates to ManifestViewer

ComponentBrowser shrinks from ~1700 to ~500 lines. Mesh loading,
tree callbacks, click-select, section plane, measure tool all
handled by ManifestViewer. Component-specific features (specs panel,
markers, steps mode) remain."
```

---

## Task 4: Change Viewport3D default camera to orthographic

**Files:**
- Modify: `viewer/viewport3d.ts`
- Modify: `viewer/viewer.ts` (bot path — may need to explicitly request perspective for sim)

- [ ] **Step 1: Change default camera type**

In `Viewport3D` constructor, change the default from `'perspective'` to `'orthographic'`. The bot viewer's `viewer.ts` currently passes `cameraType: 'perspective'` explicitly — if ManifestViewer now creates the viewport with orthographic, the bot Design tab will get orthographic too (which is the desired default per spec). The Sim tab creates its own viewport — verify it still gets perspective if needed.

- [ ] **Step 2: Verify camera toggle works in both modes**

Viewport3D already has ortho/perspective toggle buttons. Verify they work when starting from orthographic.

- [ ] **Step 3: Run lint + verify both viewers**

- [ ] **Step 4: Commit**

```bash
git add viewer/viewport3d.ts viewer/viewer.ts
git commit -m "fix: default camera to orthographic, perspective available via toggle

Orthographic is the better default for CAD inspection. Users can
switch to perspective via the existing toggle button."
```

---

## Task 5: Cleanup and integration testing

**Files:**
- Modify: various (cleanup dead code)

- [ ] **Step 1: Remove dead imports and unused code**

Grep for any remaining references to removed functions/variables in both design-viewer.ts and component-browser.ts. Remove unused imports.

- [ ] **Step 2: Run full validation**

Run: `make lint && make validate`
Verify 3 pre-existing test failures are the only failures.

- [ ] **Step 3: Manual integration testing**

Test matrix:
- `?bot=wheeler_arm` — tree, click-select, section, measure, view presets, camera toggle
- `?bot=so101_arm` — same checks with a different bot
- `?component=STS3215` — tree in left panel, specs in right panel, design layers, insertion channels, section, measure, steps mode, markers, click-select
- `?component=BEC5V` — multi-material body, fasteners, section caps
- `?component=RaspberryPiZero2W` — multi-material, mounting points
- `?cadsteps=component:STS3215` — CAD steps debugger still works
- Quick switcher (Cmd+K) — navigates between components

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "cleanup: remove dead code after viewer unification

Both viewers now use ManifestViewer as their core pipeline.
design-viewer.ts and component-browser.ts are thin wrappers."
```

---

## Verification Summary

After all tasks complete:
1. `manifest-viewer.ts` exists and owns: manifest→scene→meshes→tree→click-select→section→measure
2. `design-viewer.ts` is ~50 lines: fetch manifest, call ManifestViewer, return context
3. `component-browser.ts` is ~500 lines: fetch manifest, call ManifestViewer, specs panel, markers, steps mode
4. Both `?bot=X` and `?component=X` use the same tree-panel + side-panel layout
5. Section plane, measure tool, view presets work in both modes
6. Orthographic camera is default, perspective toggle available
7. Click-to-select works in both modes
8. Steps mode still works for components
9. `make lint && make validate` pass clean
