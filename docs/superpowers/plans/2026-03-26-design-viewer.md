# Design Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MuJoCo-based bot viewer with a Three.js Design Viewer that provides per-mesh visibility, design layers, and material support — alongside a Sim tab that preserves the existing MuJoCo functionality.

**Architecture:** Two viewer tabs sharing a single Viewport3D. The Design tab loads meshes directly from the API (no MuJoCo), builds a SceneTree for per-mesh visibility control, and renders the component tree with Fusion 360-style ● / ○ toggles. The Sim tab wraps the existing bot-viewer.ts with a pause/resume lifecycle.

**Tech Stack:** TypeScript, Three.js, STLLoader, existing Viewport3D, Python/FastAPI (server mesh endpoints)

**Spec:** `docs/superpowers/specs/2026-03-26-design-viewer-design.md`

**Run after changes:** `make lint` (ruff + biome + tsc), `make validate` (tests + renders)

**Quaternion convention:** The manifest uses MuJoCo's wxyz convention `[w, x, y, z]`. Three.js Quaternion uses xyzw. Convert with `quat.set(manifest[1], manifest[2], manifest[3], manifest[0])`. Verify this against `botcad/skeleton.py`'s `world_quat` property and `botcad/geometry.py`'s quaternion utilities.

---

### Task 1: Manifest Enhancements — Complete pos/quat for all entries + design layer metadata

**Files:**
- Modify: `botcad/emit/viewer.py`
- Test: `make validate` (manifest snapshot comparison)

The Design Viewer needs world-frame positioning for every mesh. Currently bodies have no pos/quat, comp parts are missing quat, horn parts have neither, fastener parts have neither (except on the wire-fastener-viz branch).

- [ ] **Step 1: Read current emit_viewer_manifest**

Read `botcad/emit/viewer.py` and map which entries have pos/quat:
- Bodies: NO pos/quat
- Servo parts: YES pos + quat
- Comp parts (camera, Pi, battery): YES pos, NO quat
- Horn parts: NO pos/quat
- Fastener parts: NO pos/quat
- Wire parts: NO pos/quat

- [ ] **Step 2: Add pos/quat to body entries**

In the body entry construction, add:
```python
"pos": _round_vec(body.world_pos),
"quat": _round_vec(body.world_quat),
```
`body.world_pos` and `body.world_quat` are computed by `bot.solve()`.

- [ ] **Step 3: Add quat to comp_* part entries**

Comp parts have `pos` but not `quat`. The quat comes from mount placement rotation. Look at how servo parts emit their quat (joint's `solved_servo_quat`) for reference. For mounted components, derive the quat from the mount's `rotate_z` + face rotation, or emit the identity quat if the mount rotation is already baked into the STL.

**Important:** Check whether component STLs already have mount rotation baked in (via `_apply_mount_rotation` in `build_cad`). If so, the manifest quat should be identity and `pos` is the body-local offset. The body's world transform then places everything correctly.

- [ ] **Step 4: Add pos/quat to horn part entries**

Horn meshes are oriented by `_orient_z_to_axis(horn, joint.axis)` in the server. The manifest needs the joint's position + orientation for the viewer to place them. Use the joint's solved position/quat.

- [ ] **Step 5: Add pos/quat to fastener part entries**

Fastener parts need body-local position and axis derived from mounting point geometry. For each fastener context:
- Servo ear screws: transform through `joint.solved_servo_quat` + `joint.solved_servo_center`
- Horn screws: same transform chain
- Component mount screws: transform through `mount.rotate_point()` + `mount.resolved_pos`

- [ ] **Step 6: Add design layer metadata to manifest**

Add a `design_layers` array to each joint entry in the manifest:
```python
"design_layers": [
    {"kind": "bracket", "mesh": f"bracket_{joint.name}.stl", "parent_body": body.name},
    {"kind": "coupler", "mesh": f"coupler_{joint.name}.stl", "parent_body": body.name},
    {"kind": "clearance", "mesh": f"clearance_{joint.name}.stl", "parent_body": body.name},
    {"kind": "insertion", "mesh": f"insertion_{joint.name}.stl", "parent_body": body.name},
]
```
These are positioned using the parent body's world transform (meshes are in body-local frame).

- [ ] **Step 7: Run make lint && make validate**

Check a manifest JSON to confirm all entries have pos/quat and design_layers appear on joints.

- [ ] **Step 8: Commit**

```
feat: manifest emits pos/quat for all entries + design layer metadata
```

---

### Task 2: Server — On-demand manifest endpoint + design layer mesh handlers

**Files:**
- Modify: `mindsim/server.py`
- Modify: `botcad/emit/viewer.py`

- [ ] **Step 1: Refactor emit_viewer_manifest to return dict**

Extract manifest building into `build_viewer_manifest(bot: Bot) -> dict`. The existing `emit_viewer_manifest(bot, output_dir)` becomes a thin wrapper that calls `build_viewer_manifest` and writes the result to disk.

Note: `build_viewer_manifest` only needs the `Bot` object, not the `CadModel`. The `_load_bot()` server function returns `(bot, cad)` but only `bot` is needed for the manifest.

- [ ] **Step 2: Add /api/bots/{bot}/viewer_manifest endpoint**

```python
@app.get("/api/bots/{bot}/viewer_manifest")
def get_viewer_manifest(bot: str):
    """Serve viewer manifest on demand."""
    try:
        bot_obj, _cad = _load_bot(bot)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    from botcad.emit.viewer import build_viewer_manifest
    return build_viewer_manifest(bot_obj)
```

- [ ] **Step 3: Add bracket mesh handler**

In `_generate_bot_mesh()`, add handler for `bracket_{joint_name}`:
```python
if stem.startswith("bracket_"):
    joint_name = stem.removeprefix("bracket_")
    for joint in bot.all_joints:
        if joint.name == joint_name:
            from botcad.bracket import bracket_solid
            # Build bracket in body-local frame
            solid = bracket_solid(joint.servo, body_dims=...)
            # Apply servo placement to get body-local coordinates
            solid = solid.moved(joint.solved_servo_placement)
            return _solid_to_stl_bytes(solid)
    return None
```

Reference: look at `mindsim/server.py`'s existing `_generate_solid()` function which builds bracket/coupler/clearance for the component browser. It uses `botcad/bracket.py`'s `bracket_solid()`, `coupler_solid()`, etc. The difference: for the Design Viewer these need the servo placement transform applied to put them in body-local frame.

- [ ] **Step 4: Add coupler, clearance, insertion handlers**

Same pattern as bracket. Each calls the appropriate solid builder from `botcad/bracket.py` and applies the servo placement transform.

- [ ] **Step 5: Run make lint && make validate**

Test: `curl http://localhost:8000/api/bots/wheeler_base/viewer_manifest | python -m json.tool` to verify the on-demand endpoint works.

- [ ] **Step 6: Commit**

```
feat: on-demand manifest API + design layer mesh serving
```

---

### Task 3: DesignScene data model — SceneTree, SceneNode, NodeKind

**Files:**
- Create: `viewer/design-scene.ts`
- Create: `viewer/build-scene-tree.ts`
- Create: `viewer/tests/design-scene.test.ts`

- [ ] **Step 1: Write tests for SceneTree visibility resolution**

Create `viewer/tests/design-scene.test.ts` from scratch. Test cases:

```typescript
describe('SceneTree', () => {
    // Helper to build a small tree: root > body > [comp1, comp2] > [layer1]
    function buildTestTree(): SceneTree { ... }

    // Basics
    it('node is visible by default')
    it('hidden node is not visible')
    it('unknown node returns false')

    // Cascading hide
    it('hiding parent makes children invisible')
    it('unhiding parent restores children')
    it('independently hidden child stays hidden after parent unhide')

    // Solo
    it('solo shows only the soloed subtree')
    it('solo shows ancestors on the path to root')
    it('solo hides siblings of soloed node')
    it('unsolo restores all visibility')

    // Combined
    it('hidden node inside soloed subtree stays hidden')

    // meshIds
    it('resolveVisibility works with meshIds mapping')
})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pnpm exec vitest run viewer/tests/design-scene.test.ts
```

- [ ] **Step 3: Implement NodeKind, SceneNode, SceneTree**

Create `viewer/design-scene.ts`:
- `NodeKind` enum: `Robot`, `Assembly`, `Body`, `Component`, `SubPart`, `Joint`, `Layer`
- `SceneNode` interface: `id`, `kind`, `label`, `children`, `hidden`, `parentId`, `meshIds`
- `SceneTree` class:
  - `nodes: Map<string, SceneNode>`
  - `soloedId: string | null`
  - `addNode(node)`, `getNode(id)`
  - `resolveVisibility(nodeId)` — walks ancestors for hidden + checks solo
  - `toggleHidden(nodeId)` — flip hidden flag
  - `solo(nodeId)` / `unsolo()` — set/clear soloedId
- `DesignScene` class:
  - `tree: SceneTree`
  - `meshes: Map<string, THREE.Mesh>`
  - `meshToNode: Map<string, string>` — mesh UUID → node ID
  - `registerMesh(nodeId, mesh)` — adds mesh to maps + updates node's meshIds
  - `syncVisibility()` — iterate meshes, set `visible` from `tree.resolveVisibility()`

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Implement buildSceneTree**

Create `viewer/build-scene-tree.ts`. Builds SceneTree from manifest JSON:
- Robot root node from `manifest.bot_name`
- Body nodes (if assemblies exist, nest under Assembly nodes)
- Component nodes for mounted parts (non-servo, non-horn, non-fastener)
- SubPart nodes for horns, fasteners (grouped: "6× M3 SHC")
- Servo Component nodes per joint
- Joint nodes linking parent body to child body
- Layer nodes for design layers from `joint.design_layers[]` — created with `hidden: true`
- Wire group node per body — created with `hidden: true`

`meshIds` is empty at build time — populated when meshes are loaded.

- [ ] **Step 6: Run all tests, commit**

```
feat: DesignScene data model with SceneTree visibility resolution
```

---

### Task 4: Component Tree UI — ● / ○ toggles + Solo

**Files:**
- Modify: `viewer/component-tree.ts`
- Test: Visual verification in browser (after Task 5 wires it up)

This task modifies the component tree rendering. It compiles and lints but cannot be functionally tested until Task 5 wires it to the Design Viewer. That's OK — tsc + biome catch structural issues.

- [ ] **Step 1: Replace eye/target icons with right-aligned ● / ○**

In `component-tree.ts`, find the visibility action icons section (eye and target buttons in `_buildNode`). Replace with a single right-aligned circle indicator:
```typescript
// Right-aligned visibility dot
const dot = document.createElement('span');
dot.className = 'vis-dot';
dot.style.cssText = 'margin-left: auto; width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; cursor: pointer;';
dot.style.background = '#58a6ff'; // visible state
dot.addEventListener('click', (e) => {
    e.stopPropagation();
    if (this.onToggleNodeHidden) this.onToggleNodeHidden(nodeId);
});
header.appendChild(dot);
```

Remove the old eye icon (`tree-vis-eye`) and target icon (`tree-vis-target`) elements.

- [ ] **Step 2: Add new callbacks to ComponentTreeOptions**

```typescript
onToggleNodeHidden?: (nodeId: string) => void;
onSolo?: (nodeId: string) => void;
onUnsolo?: () => void;
```

Remove old `onToggleVisibility`, `onIsolate`, `onShowAll` callbacks.

- [ ] **Step 3: Add right-click Solo**

Add `contextmenu` event listener on each tree node header:
```typescript
header.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (this.onSolo) this.onSolo(nodeId);
});
```

- [ ] **Step 4: Add Escape to un-solo**

In the ComponentTree constructor, listen for Escape:
```typescript
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && this.onUnsolo) this.onUnsolo();
});
```

- [ ] **Step 5: Add updateFromDesignScene method**

```typescript
updateFromDesignScene(tree: SceneTree): void {
    const allNodes = this._treeRoot?.querySelectorAll<HTMLElement>('.tree-node[data-node-id]');
    if (!allNodes) return;
    for (const domNode of allNodes) {
        const nodeId = domNode.dataset.nodeId!;
        const visible = tree.resolveVisibility(nodeId);
        const dot = domNode.querySelector('.vis-dot') as HTMLElement;
        if (dot) {
            dot.style.background = visible ? '#58a6ff' : 'transparent';
            dot.style.border = visible ? 'none' : '2px solid #484f58';
        }
        domNode.style.opacity = visible ? '1' : '0.45';
    }
}
```

- [ ] **Step 6: Run make lint (tsc + biome)**

This verifies types compile. Functional testing happens in Task 7.

- [ ] **Step 7: Commit**

```
feat: component tree with ●/○ visibility toggles and right-click Solo
```

---

### Task 5: Design Viewer — Three.js scene loader

**Files:**
- Create: `viewer/design-viewer.ts`

The core new file. Loads meshes from the API, positions them, wires SceneTree for per-mesh visibility.

- [ ] **Step 1: Scaffold design-viewer.ts**

```typescript
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { DesignScene } from './design-scene.ts';
import { buildSceneTree } from './build-scene-tree.ts';
import { ComponentTree } from './component-tree.ts';
import type { Viewport3D } from './viewport3d.ts';

export interface DesignViewerContext {
    scene: DesignScene;
    tree: ComponentTree;
    viewport: Viewport3D;
    syncVisibility(): void;
}

export async function initDesignViewer(
    botName: string,
    viewport: Viewport3D,
    treePanelEl: HTMLElement,
): Promise<DesignViewerContext> { ... }
```

- [ ] **Step 2: Fetch manifest and build SceneTree**

```typescript
const resp = await fetch(`/api/bots/${botName}/viewer_manifest`);
const manifest = await resp.json();
const scene = new DesignScene();
buildSceneTree(scene.tree, manifest);
```

- [ ] **Step 3: Load body meshes**

```typescript
const stlLoader = new STLLoader();
const bodyMaterial = new THREE.MeshPhysicalMaterial({ color: 0xc0c8d0, roughness: 0.8 });

for (const body of manifest.bodies) {
    const buf = await fetch(`/api/bots/${botName}/meshes/${body.mesh}`).then(r => r.arrayBuffer());
    const geometry = stlLoader.parse(buf);
    geometry.computeVertexNormals();
    const mesh = new THREE.Mesh(geometry, bodyMaterial.clone());
    mesh.position.set(body.pos[0], body.pos[1], body.pos[2]);
    // Convert wxyz → xyzw quaternion
    mesh.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    viewport.scene.add(mesh);
    scene.registerMesh(`body:${body.name}`, mesh);
}
```

- [ ] **Step 4: Load component part meshes**

For each part in manifest (by category: servo, camera, compute, battery, wheel, horn):
- Fetch STL from `/api/bots/${botName}/meshes/${part.mesh}`
- Create `MeshPhysicalMaterial` from manifest `materials` dict (if multi-material, load per-material STLs)
- Position using part's `pos`/`quat` (same wxyz → xyzw conversion)
- Add to viewport scene
- Register in SceneTree: `scene.registerMesh(`part:${part.id}`, mesh)`

- [ ] **Step 5: Wire up visibility sync**

```typescript
function syncVisibility(scene: DesignScene) {
    scene.syncVisibility(); // iterates meshes, sets .visible from tree
}
```

- [ ] **Step 6: Build component tree and wire callbacks**

```typescript
const tree = new ComponentTree(treePanelEl, {
    onToggleNodeHidden: (nodeId) => {
        scene.tree.toggleHidden(nodeId);
        syncVisibility(scene);
        tree.updateFromDesignScene(scene.tree);
    },
    onSolo: (nodeId) => {
        scene.tree.solo(nodeId);
        syncVisibility(scene);
        tree.updateFromDesignScene(scene.tree);
    },
    onUnsolo: () => {
        scene.tree.unsolo();
        syncVisibility(scene);
        tree.updateFromDesignScene(scene.tree);
    },
});
tree.build(); // renders from manifest
```

- [ ] **Step 7: Add lazy loading for design layers**

When user toggles a Layer node visible for the first time:
```typescript
async function lazyLoadLayer(nodeId: string, scene: DesignScene, ...) {
    const node = scene.tree.getNode(nodeId);
    if (!node || node.kind !== NodeKind.Layer || node.meshIds.length > 0) return;
    // Derive mesh stem from node data (e.g., "bracket_left_wheel")
    const resp = await fetch(`/api/bots/${botName}/meshes/${meshStem}.stl`);
    const geometry = stlLoader.parse(await resp.arrayBuffer());
    // ... create mesh, position using parent body transform, register in SceneTree
}
```

- [ ] **Step 8: Frame camera on loaded geometry**

After all eager meshes load, compute bounding box and call `viewport.frameOnBox()`.

- [ ] **Step 9: Run make lint**

- [ ] **Step 10: Commit**

```
feat: Design Viewer — Three.js scene loader with per-mesh visibility
```

---

### Task 6: Tab System — Design + Sim + Component mode

**Files:**
- Modify: `viewer/viewer.ts`
- Modify: `viewer/bot-viewer.ts`

- [ ] **Step 1: Refactor viewer.ts routing for ?bot=X**

When `?bot=X`:
1. Create shared `Viewport3D` (perspective camera)
2. Create Design and Sim tab buttons in the nav bar (alongside existing mode buttons)
3. Default to Design tab
4. `initDesignViewer(botName, viewport, treePanelEl)` immediately
5. Lazy-load Sim (`initBotViewer`) only when Sim tab clicked for the first time

- [ ] **Step 2: Route ?component=X through Design Viewer**

When `?component=X`:
1. Create `Viewport3D` (orthographic camera, matching current component browser)
2. Build a minimal manifest for the single component (or use a dedicated `/api/component/{name}/manifest` endpoint)
3. `initDesignViewer(componentName, viewport, treePanelEl)` — same viewer, component scope
4. No Sim tab for component mode

This replaces the `component-browser.ts` import. The component browser's layer system is now handled by the SceneTree's Layer nodes.

- [ ] **Step 3: Add tab switching for ?bot=X**

```typescript
let simViewer: SimViewerContext | null = null;
let designActive = true;

function switchTab(tab: 'design' | 'sim') {
    if (tab === 'sim' && !simViewer) {
        // First time: lazy-load MuJoCo viewer
        simViewer = await initBotViewer(botName, viewport);
    }
    if (tab === 'design') {
        simViewer?.pause();
        // Show Design scene in viewport
        viewport.scene = designViewer.scene.threeScene;
        designActive = true;
    } else {
        // Show MuJoCo scene in viewport
        viewport.scene = simViewer.scene;
        simViewer.resume();
        designActive = false;
    }
}
```

- [ ] **Step 4: Wrap bot-viewer.ts with pause/resume lifecycle**

Add to the bot viewer's animation loop:
```typescript
let paused = false;
function pause() { paused = true; }
function resume() { paused = false; requestAnimationFrame(loop); }

function loop() {
    if (paused) return;
    // ... existing animation frame logic
    requestAnimationFrame(loop);
}
```

Export `pause()` and `resume()` as part of the Sim viewer context.

- [ ] **Step 5: Run make lint && make validate**

- [ ] **Step 6: Commit**

```
feat: Design/Sim tab system + component mode via Design Viewer
```

---

### Task 7: Integration Testing + Polish

No new files — bug fixes and visual verification across the full system.

- [ ] **Step 1: Test full loading flow**

Load `?bot=wheeler_base` in Design tab. Use Playwright or manual browser testing:
- All bodies visible and correctly positioned
- All components visible with correct materials
- Component tree shows correct hierarchy with ● indicators
- No overlapping or floating parts

- [ ] **Step 2: Test per-mesh visibility**

- Click ● on a component (e.g., Battery) → mesh disappears, ○ indicator
- Click ○ → mesh reappears, ● indicator
- Click ● on a body → body + all mounted components disappear
- Unhide body → components restore to their previous state

- [ ] **Step 3: Test Solo**

- Right-click a servo node → only servo + its children (bracket, horn, fasteners) visible
- Everything else hidden
- Press Escape → all restored

- [ ] **Step 4: Test design layers**

- Expand a servo in the tree
- Click ○ on Bracket → STL fetches, bracket mesh appears
- Click ○ on Clearance Envelope → envelope mesh appears
- Click ● on Bracket → mesh disappears

- [ ] **Step 5: Test tab switching**

- Click Sim tab → MuJoCo loads (first time), joints mode works
- Click Design tab → Design scene restored, camera position same
- Visibility state independent between tabs

- [ ] **Step 6: Test component mode**

- Load `?component=STS3215` → single servo with design layers
- Layer toggles work (bracket, coupler, clearance)
- Section cutter works

- [ ] **Step 7: Run make lint && make validate**

- [ ] **Step 8: Final commit**

```
chore: integration fixes for Design Viewer
```
