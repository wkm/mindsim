# Component Viewer: Manifest + Tree Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the component viewer's ad-hoc layer system with `ComponentTree` driven by a server-generated `ViewerManifest`, unifying the component and bot viewer code paths.

**Architecture:** New server endpoint `/api/components/{name}/manifest` generates a `ViewerManifest`-compatible JSON for a single component (one body, one mount, parts for fasteners/wires, materials dict). The component browser drops `LAYER_META`, `_loadLayer`, `_loadFasteners`, `_loadWires`, and layer checkboxes, replacing them with a `ComponentTree` instance that controls visibility via the same callbacks used by the bot viewer's `DesignViewer`.

**Tech Stack:** Python/FastAPI (server), TypeScript/Three.js (viewer), existing `ComponentTree`, `ViewerManifest` types, `DesignScene`/`SceneTree` data model.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `mindsim/server.py` | Modify | Add `/api/components/{name}/manifest` endpoint |
| `viewer/component-browser.ts` | Modify | Remove layer system, integrate ComponentTree + DesignScene |
| `viewer/component-tree.ts` | Modify (minor) | Ensure single-component manifests render cleanly (no joints = no sub-assemblies) |
| `tests/test_component_manifest.py` | Create | Test manifest endpoint for representative components |

---

## Task 1: Server — Component Manifest Endpoint

**Files:**
- Create: `tests/test_component_manifest.py`
- Modify: `mindsim/server.py` (after line ~903, before the render-svg endpoint)

This task adds `GET /api/components/{name}/manifest` that returns a `ViewerManifest`-shaped JSON for a single component.

- [ ] **Step 1: Write failing test for the manifest endpoint**

```python
# tests/test_component_manifest.py
"""Tests for the component manifest endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from mindsim.server import app
    return TestClient(app)


def test_component_manifest_structure(client):
    """Manifest for a multi-material component has correct shape."""
    resp = client.get("/api/components/BEC5V/manifest")
    assert resp.status_code == 200
    m = resp.json()

    # Top-level fields
    assert m["bot_name"] == "BEC5V"
    assert isinstance(m["bodies"], list)
    assert isinstance(m["joints"], list)
    assert isinstance(m["mounts"], list)
    assert isinstance(m["parts"], list)
    assert isinstance(m["materials"], dict)

    # Single body, identity pose
    assert len(m["bodies"]) == 1
    body = m["bodies"][0]
    assert body["name"] == "BEC5V"
    assert body["parent"] is None
    assert body["role"] == "structure"
    assert body["mesh"] == "body"
    assert body["pos"] == [0, 0, 0]
    assert body["quat"] == [1, 0, 0, 0]

    # No joints for a standalone component
    assert m["joints"] == []

    # Self-referential mount
    assert len(m["mounts"]) == 1
    mount = m["mounts"][0]
    assert mount["body"] == "BEC5V"
    assert mount["component"] == "BEC5V"
    assert mount["category"] == "component"


def test_component_manifest_multi_material(client):
    """Multi-material components include meshes array and materials dict."""
    resp = client.get("/api/components/BEC5V/manifest")
    m = resp.json()
    mount = m["mounts"][0]

    # Should have meshes array for multi-material rendering
    assert "meshes" in mount
    assert len(mount["meshes"]) > 0
    for mesh_entry in mount["meshes"]:
        assert "file" in mesh_entry
        assert "material" in mesh_entry
        # Material must exist in the materials dict
        assert mesh_entry["material"] in m["materials"]

    # Each material has required fields
    for mat_name, mat in m["materials"].items():
        assert "color" in mat and len(mat["color"]) == 3
        assert "metallic" in mat
        assert "roughness" in mat
        assert "opacity" in mat


def test_component_manifest_fasteners(client):
    """Components with mounting points include fastener parts."""
    # BEC5V has mounting points → should have fastener parts
    resp = client.get("/api/components/BEC5V/manifest")
    m = resp.json()

    fasteners = [p for p in m["parts"] if p["category"] == "fastener"]
    assert len(fasteners) > 0
    for f in fasteners:
        assert "id" in f
        assert "name" in f
        assert f["parent_body"] == "BEC5V"
        assert "mesh" in f
        assert "pos" in f
        assert "quat" in f


def test_component_manifest_wires(client):
    """Components with wire ports include wire parts."""
    # STS3215 servo has wire ports
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    wires = [p for p in m["parts"] if p["category"] == "wire"]
    assert len(wires) > 0
    for w in wires:
        assert "id" in w
        assert "name" in w
        assert w["parent_body"] == "STS3215"
        assert "bus_type" in w
        assert "pos" in w


def test_component_manifest_404(client):
    """Unknown component returns 404."""
    resp = client.get("/api/components/nonexistent/manifest")
    assert resp.status_code == 404


def test_servo_manifest_has_layers(client):
    """Servo components include body layers (bracket, horn, etc.) as additional bodies."""
    resp = client.get("/api/components/STS3215/manifest")
    m = resp.json()

    # Servo manifest should still have one structural body
    assert len(m["bodies"]) == 1
    # The mount should reference the servo component
    assert len(m["mounts"]) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_component_manifest.py -v`
Expected: FAIL — 404 because the endpoint doesn't exist yet.

- [ ] **Step 3: Implement the manifest endpoint**

Add to `mindsim/server.py`, after the `/api/components/{name}/wires` endpoint (around line 903):

```python
@app.get("/api/components/{name:path}/manifest")
def get_component_manifest(name: str):
    """Return a ViewerManifest-compatible JSON for a single component.

    Shape matches the bot viewer manifest so ComponentTree can consume it
    directly. One body (identity pose), one self-referential mount,
    fasteners/wires as parts, and a materials dict.
    """
    if name not in _component_registry:
        raise HTTPException(404, f"Unknown component: {name}")

    _factory, comp, category = _component_registry[name]

    from botcad.component import get_component_meta
    from botcad.connectors import connector_spec
    from botcad.geometry import rotation_between

    meta = get_component_meta(comp.kind)

    # ── Body: single structural body at origin ──
    comp_color = (
        list(comp.default_material.color[:3])
        if comp.default_material
        else [0.541, 0.608, 0.659]
    )
    body = {
        "name": name,
        "parent": None,
        "role": "structure",
        "mesh": "body",
        "pos": [0, 0, 0],
        "quat": [1, 0, 0, 0],
        "color": comp_color,
    }

    # ── Mount: self-referential ──
    mount: dict = {
        "body": name,
        "label": name,
        "component": name,
        "category": "component",
        "mesh": "body",
        "pos": [0, 0, 0],
        "quat": [1, 0, 0, 0],
    }

    # Multi-material meshes
    materials: dict[str, dict] = {}
    if meta.multi_material_emitter is not None:
        try:
            mm_result = meta.multi_material_emitter(comp)
            if mm_result is not None:
                meshes = []
                for mp in mm_result.material_programs:
                    mat = mp.material
                    mat_key = f"mat__{mat.name}"
                    meshes.append({"file": mat_key, "material": mat.name})
                    materials[mat.name] = {
                        "color": list(mat.color[:3]),
                        "metallic": mat.metallic,
                        "roughness": mat.roughness,
                        "opacity": getattr(mat, "opacity", 1.0),
                    }

                    # Ensure per-material STL is cached
                    cache_key = (comp.name, mat_key)
                    with _stl_cache_lock:
                        if cache_key not in _stl_cache:
                            from botcad.shapescript.backend_occt import OcctBackend

                            result = OcctBackend().execute(mp.program)
                            solid = result.shapes[mp.program.output_ref.id]
                            _stl_cache[cache_key] = _solid_to_stl_bytes(solid)

                mount["meshes"] = meshes
        except Exception:
            pass

    # Add component default material if not already in materials dict
    if comp.default_material and comp.default_material.name not in materials:
        mat = comp.default_material
        materials[mat.name] = {
            "color": list(mat.color[:3]),
            "metallic": mat.metallic,
            "roughness": mat.roughness,
            "opacity": getattr(mat, "opacity", 1.0),
        }

    # ── Parts: fasteners from mounting points ──
    parts: list[dict] = []
    for i, mp in enumerate(comp.mounting_points):
        d_key = f"{mp.diameter:.4f}"
        screw_mesh = f"_screw_{d_key}"

        # Ensure screw STL is cached
        cache_key = (screw_mesh, "body")
        with _stl_cache_lock:
            if cache_key not in _stl_cache:
                try:
                    from botcad.emit.cad import screw_solid

                    stl_bytes = _solid_to_stl_bytes(screw_solid(mp.diameter))
                    _stl_cache[cache_key] = stl_bytes
                except Exception:
                    pass

        parts.append({
            "id": f"fastener_{name}_{mp.label}",
            "name": f"M{mp.diameter * 1000:.0f} screw",
            "category": "fastener",
            "parent_body": name,
            "mesh": screw_mesh,
            "pos": list(mp.pos),
            "quat": _axis_to_quat(mp.axis),
        })

    # ── Parts: wires from wire ports ──
    bus_colors = {
        "uart_half_duplex": [0.20, 0.60, 0.86, 1.0],
        "csi": [0.40, 0.73, 0.42, 1.0],
        "power": [0.90, 0.30, 0.25, 1.0],
        "usb": [0.55, 0.35, 0.75, 1.0],
    }

    # Ensure shared wire stub STL is generated
    stub_cache_key = ("_wire_stub", "body")
    with _stl_cache_lock:
        if stub_cache_key not in _stl_cache:
            from botcad.shapescript.backend_occt import OcctBackend
            from botcad.shapescript.program import ShapeScript

            prog = ShapeScript()
            stub = prog.cylinder(0.0015, 0.025, tag="wire_stub")
            prog.output_ref = stub
            result = OcctBackend().execute(prog)
            _stl_cache[stub_cache_key] = _solid_to_stl_bytes(result.shapes[stub.id])

    for wp in comp.wire_ports:
        if not wp.connector_type:
            continue
        try:
            cspec = connector_spec(wp.connector_type)
        except KeyError:
            continue

        exit_dir = cspec.wire_exit_direction
        quat = rotation_between((0.0, 0.0, 1.0), exit_dir)
        half_len = 0.0125
        center = (
            wp.pos[0] + exit_dir[0] * half_len,
            wp.pos[1] + exit_dir[1] * half_len,
            wp.pos[2] + exit_dir[2] * half_len,
        )
        color = bus_colors.get(str(wp.bus_type), [0.53, 0.53, 0.53, 1.0])

        parts.append({
            "id": f"wire_{name}_{wp.label}",
            "name": wp.label,
            "category": "wire",
            "parent_body": name,
            "mesh": "_wire_stub",
            "pos": list(center),
            "quat": [quat[0], quat[1], quat[2], quat[3]],
            "bus_type": str(wp.bus_type),
            "connector_type": wp.connector_type,
            "color": color,
        })

        # Connector housing
        conn_key = f"_connector_{wp.connector_type}"
        cache_key = (conn_key, "body")
        with _stl_cache_lock:
            if cache_key not in _stl_cache:
                try:
                    from botcad.connectors import connector_solid

                    solid = connector_solid(cspec)
                    _stl_cache[cache_key] = _solid_to_stl_bytes(solid)
                except Exception:
                    pass

        conn_quat = rotation_between((0.0, 0.0, 1.0), cspec.mating_direction)
        parts.append({
            "id": f"connector_{name}_{wp.label}",
            "name": f"{wp.label} ({cspec.label})",
            "category": "wire",
            "parent_body": name,
            "mesh": conn_key,
            "pos": list(wp.pos),
            "quat": [conn_quat[0], conn_quat[1], conn_quat[2], conn_quat[3]],
            "bus_type": str(wp.bus_type),
            "color": color,
        })

    return {
        "bot_name": name,
        "bodies": [body],
        "joints": [],
        "mounts": [mount],
        "parts": parts,
        "materials": materials,
        "assemblies": [],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_component_manifest.py -v`
Expected: All PASS.

- [ ] **Step 5: Run full lint + validate**

Run: `make lint && make validate`
Expected: Clean.

- [ ] **Step 6: Commit**

```bash
git add mindsim/server.py tests/test_component_manifest.py
git commit -m "feat: add /api/components/{name}/manifest endpoint

Returns ViewerManifest-shaped JSON for a single component — one body,
one self-referential mount, fasteners/wires as parts, materials dict.
This lets the component browser use ComponentTree instead of LAYER_META."
```

---

## Task 2: Component Browser — Integrate ComponentTree + DesignScene

**Files:**
- Modify: `viewer/component-browser.ts`

This is the main refactor: replace the layer system with ComponentTree-driven visibility. The component browser fetches the manifest, builds a `SceneTree` + `DesignScene`, creates a `ComponentTree`, and loads meshes on demand via tree callbacks.

- [ ] **Step 1: Add imports and remove LAYER_META**

In `component-browser.ts`:

1. Add new imports at the top:
```typescript
import { ComponentTree } from './component-tree.ts';
import { buildSceneTree } from './build-scene-tree.ts';
import { DesignScene, NodeKind } from './design-scene.ts';
import type { ViewerManifest, ManifestMount, ManifestPart } from './manifest-types.ts';
```

2. Remove lines 37-57 (`LAYER_META` constant and `ALL_LAYER_IDS`).

3. Remove `layerGroups` field from the class and its initialization in the constructor.

4. Add new fields:
```typescript
  _manifest: ViewerManifest | null;
  _designScene: DesignScene | null;
  _componentTree: ComponentTree | null;
  _meshGroups: Record<string, THREE.Group>;
```

Initialize them in the constructor:
```typescript
    this._manifest = null;
    this._designScene = null;
    this._componentTree = null;
    this._meshGroups = {};
```

- [ ] **Step 2: Replace _setupViewport layer group creation**

Remove the layer group creation loop (lines 157-162):
```typescript
    // OLD — remove this:
    for (const id of ALL_LAYER_IDS) {
      const g = this.viewport.addGroup(id);
      this.layerGroups[id] = g;
    }
```

Remove the `setSectionCapColorFn` callback that references LAYER_META (lines 164-179). It will be re-added in the new `loadComponent`.

- [ ] **Step 3: Replace loadComponent**

Replace the `loadComponent` method. The new flow:
1. Fetch manifest from `/api/components/{name}/manifest`
2. Build SceneTree + DesignScene from the manifest
3. Clear old viewport groups, create new ones per tree node
4. Load the body mesh (always visible)
5. Build ComponentTree in the tree panel area
6. Wire up callbacks: toggle hidden, solo/unsolo, category toggle
7. Set default visibility (fasteners/wires hidden)
8. Register section cap color callback using manifest materials

```typescript
  async loadComponent(name: string) {
    const comp = this.components.find((c) => c.name === name);
    if (!comp) return;
    this.currentComponent = comp;

    document.getElementById('bot-name').textContent = name;
    document.getElementById('mode-tabs').style.display = 'none';

    // Clear measurements, section, steps mode
    this.viewport._meas.clearAll();
    this._resetSection();
    if (this.stepsMode) {
      this.stepsMode = false;
      clearGroup(this.stepsGroup);
      clearGroup(this.stepsToolGroup);
      this.stepsGroup.visible = false;
      this.stepsToolGroup.visible = false;
      this.stepsData = null;
      this._stepsHasFramed = false;
      const btn = document.getElementById('steps-toggle');
      if (btn) btn.classList.remove('active');
    }

    // Clear previous mesh groups (Viewport3D has no removeGroup — clear + remove from scene)
    for (const [id, group] of Object.entries(this._meshGroups)) {
      this.viewport.clearGroup(group);
      this.scene.remove(group);
    }
    this._meshGroups = {};

    // Dispose previous tree
    if (this._componentTree) {
      this._componentTree.dispose();
      this._componentTree = null;
    }

    // Fetch manifest
    const resp = await fetch(`/api/components/${name}/manifest`);
    if (!resp.ok) return;
    this._manifest = await resp.json() as ViewerManifest;

    // Build scene tree + design scene
    const sceneTree = buildSceneTree(this._manifest);
    this._designScene = new DesignScene(sceneTree);

    // Create viewport groups for body + mount + parts
    const bodyGroup = this.viewport.addGroup(`body:${name}`);
    this._meshGroups[`body:${name}`] = bodyGroup;

    // Load body mesh (always visible)
    const mount = this._manifest.mounts?.[0];
    if (mount?.meshes && mount.meshes.length > 0) {
      // Multi-material
      await this._loadMultiMaterialMount(name, mount, bodyGroup);
    } else {
      // Single material body
      const entityColor = comp.color
        ? new THREE.Color(comp.color[0], comp.color[1], comp.color[2])
        : new THREE.Color(0.541, 0.608, 0.659);
      await this._addSTLMesh(name, 'body', tintColor(entityColor), {}, bodyGroup);
    }

    // Register body meshes with design scene.
    // buildSceneTree creates: body node as `body:{name}`, mount as `mount:{name}:{name}`.
    // Register under the mount node since that's the component container.
    const mountNodeId = `mount:${name}:${name}`;
    const bodyNodeId = `body:${name}`;
    // Use mount node if it exists (multi-material), body node otherwise
    const regNodeId = this._designScene.tree.getNode(mountNodeId) ? mountNodeId : bodyNodeId;
    for (const child of bodyGroup.children) {
      this._designScene.registerMesh(regNodeId, child as THREE.Mesh);
    }

    // Update side panel (specs section — no layer checkboxes)
    this._updateSidePanel(comp);

    // Build ComponentTree in a container
    const treeContainer = this._getOrCreateTreeContainer();
    this._componentTree = new ComponentTree(treeContainer, this._manifest,
      (nodeId) => { /* select callback — no-op for now */ },
      {
        onToggleNodeHidden: (nodeId) => this._onTreeToggle(nodeId),
        onSolo: (nodeId) => this._onTreeSolo(nodeId),
        onUnsolo: () => this._onTreeUnsolo(),
        onCategoryToggle: (category, visible) => this._onCategoryToggle(category, visible),
        onShapeScript: (url) => {
          // Route ShapeScript links to component endpoint instead of bot endpoint
          window.location.href = url.replace(/\?cadsteps=/, `?component=${name}&cadsteps=`);
        },
      },
    );
    this._componentTree.build();

    // Set fasteners and wires hidden by default.
    // SceneTree.nodes is a Map — use .values().
    // Fasteners are SubPart nodes; wire groups are Component nodes with `wire-group:` prefix.
    for (const node of this._designScene.tree.nodes.values()) {
      if (node.kind === NodeKind.SubPart || node.id.startsWith('wire-group:')) {
        node.hidden = true;
      }
    }
    this._designScene.syncVisibility();
    this._componentTree.updateFromDesignScene(this._designScene.tree);

    // Section cap color from manifest materials
    this.viewport.setSectionCapColorFn((groupName: string) => {
      if (!this._manifest) return null;
      // Try to find a matching material from the manifest
      const materials = this._manifest.materials ?? {};
      // Body groups use component color
      const comp = this.currentComponent;
      if (!comp) return 0xced9e0;
      const entityColor = new THREE.Color(comp.color[0], comp.color[1], comp.color[2]);
      return tintColor(entityColor, 0.6);
    });

    this._fitCameraToVisibleMeshes();
  }
```

- [ ] **Step 4: Add tree callback methods**

Add these new methods to the class:

```typescript
  _getOrCreateTreeContainer(): HTMLElement {
    let container = document.getElementById('component-tree-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'component-tree-container';
      // Insert at the top of the side panel
      const panel = document.getElementById('side-panel');
      panel.insertBefore(container, panel.firstChild);
    }
    container.innerHTML = '';
    return container;
  }

  async _onTreeToggle(nodeId: string) {
    if (!this._designScene) return;
    const node = this._designScene.tree.getNode(nodeId);
    if (!node) return;

    node.hidden = !node.hidden;

    // Lazy-load mesh if this is the first time making it visible
    if (!node.hidden && node.meshIds.length === 0) {
      await this._lazyLoadNodeMesh(nodeId);
    }

    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);

    // Re-fit camera and section
    const box = this._getVisibleBBox();
    this.viewport._fitOrthoFrustum(box);
    this.controls.update();
    if (this.sectionEnabled) this._updateSectionPlane();
  }

  _onTreeSolo(nodeId: string) {
    if (!this._designScene) return;
    this._designScene.tree.solo(nodeId);
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _onTreeUnsolo() {
    if (!this._designScene) return;
    this._designScene.tree.unsolo();
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _onCategoryToggle(category: string, visible: boolean) {
    if (!this._designScene) return;
    // Toggle hidden on all nodes of this category (nodes is a Map)
    for (const node of this._designScene.tree.nodes.values()) {
      const nodeCategory = this._nodeCategoryForFilter(node);
      if (nodeCategory === category) {
        node.hidden = !visible;
      }
    }
    this._designScene.syncVisibility();
    this._componentTree?.updateFromDesignScene(this._designScene.tree);
  }

  _nodeCategoryForFilter(node: any): string | null {
    // Map node kinds to filter categories.
    // Use NodeKind enum (imported from design-scene.ts) and node ID prefixes.
    if (node.kind === NodeKind.Body) return 'body';
    if (node.kind === NodeKind.SubPart) {
      if (node.id.startsWith('fastener')) return 'fastener';
    }
    // Wire groups are NodeKind.Component with `wire-group:` prefix
    if (node.id.startsWith('wire-group:')) return 'wire';
    if (node.kind === NodeKind.Component) return 'mount';
    return null;
  }
```

- [ ] **Step 5: Add lazy-load mesh method**

```typescript
  async _lazyLoadNodeMesh(nodeId: string) {
    if (!this._manifest || !this._designScene) return;
    const compName = this._manifest.bot_name;
    const allParts = this._manifest.parts ?? [];

    // Match node ID to manifest parts.
    // buildSceneTree generates these ID patterns:
    //   fastener-group:mount:{body}:{label}:{groupKey}  (fasteners on a mount)
    //   wire-group:{body}                                (all wires on a body)
    let matchingParts: ManifestPart[];

    if (nodeId.startsWith('wire-group:')) {
      // Wire group: load ALL wire parts for this body
      matchingParts = allParts.filter((p) => p.category === 'wire');
    } else if (nodeId.startsWith('fastener-group:')) {
      // Fastener group: match by group key (last segment of node ID).
      // groupKey comes from groupFasteners() in utils.ts — it's the part name.
      const segments = nodeId.split(':');
      const groupKey = segments[segments.length - 1];
      matchingParts = allParts.filter(
        (p) => p.category === 'fastener' && p.name === groupKey,
      );
    } else {
      return; // Unknown node type
    }

    if (matchingParts.length === 0) return;

    // Create or find the viewport group for this node
    let group = this._meshGroups[nodeId];
    if (!group) {
      group = this.viewport.addGroup(nodeId);
      this._meshGroups[nodeId] = group;
    }

    // Deduplicate STL URLs, batch-load geometries
    const stlUrlForPart = (part: ManifestPart) => {
      // Meshes starting with '_' are shared synthetic components (_screw_*, _connector_*, _wire_stub)
      const meshName = part.mesh;
      return meshName.startsWith('_')
        ? `/api/components/${meshName}/stl/body`
        : `/api/components/${compName}/stl/${meshName}`;
    };

    const uniqueUrls = [...new Set(matchingParts.map(stlUrlForPart))];
    const geomCache: Record<string, THREE.BufferGeometry> = {};
    await Promise.all(
      uniqueUrls.map(async (url) => {
        try {
          const resp = await fetch(url);
          if (!resp.ok) return;
          const buf = await resp.arrayBuffer();
          const geom = this.stlLoader.parse(buf);
          geom.computeVertexNormals();
          geomCache[url] = geom;
        } catch { /* skip */ }
      }),
    );

    // Place each part with its position/quaternion
    for (const part of matchingParts) {
      const url = stlUrlForPart(part);
      const srcGeom = geomCache[url];
      if (!srcGeom) continue;

      let color: THREE.Color | number;
      if (part.category === 'fastener') {
        color = tintColor(0xd4a843);
      } else if (part.color) {
        color = tintColor(new THREE.Color(part.color[0], part.color[1], part.color[2]));
      } else {
        color = tintColor(0x808080);
      }

      const mat = createMaterial(color);
      const mesh = new THREE.Mesh(srcGeom.clone(), mat);
      mesh.castShadow = true;
      if (part.pos) mesh.position.set(part.pos[0], part.pos[1], part.pos[2]);
      if (part.quat) mesh.quaternion.set(part.quat[1], part.quat[2], part.quat[3], part.quat[0]);
      group.add(mesh);

      this._designScene.registerMesh(nodeId, mesh);
    }
  }
```

- [ ] **Step 6: Update _loadMultiMaterialMount helper**

Add this method (extracted from the existing `_tryLoadMultiMaterial` but taking manifest mount data):

```typescript
  async _loadMultiMaterialMount(compName: string, mount: ManifestMount, group: THREE.Group) {
    if (!mount.meshes) return;
    const materials = this._manifest?.materials ?? {};

    await Promise.all(
      mount.meshes.map(async (meshEntry) => {
        const stlUrl = `/api/components/${compName}/stl/${meshEntry.file}`;
        try {
          const stlResp = await fetch(stlUrl);
          if (!stlResp.ok) return;
          const buf = await stlResp.arrayBuffer();
          const geometry = this.stlLoader.parse(buf);
          geometry.computeVertexNormals();

          const mat = materials[meshEntry.material];
          const color = mat
            ? new THREE.Color(mat.color[0], mat.color[1], mat.color[2])
            : new THREE.Color(0.5, 0.5, 0.5);
          addMeshWithEdges(geometry, tintColor(color), group, {
            metalness: mat?.metallic ?? 0,
            roughness: mat?.roughness ?? 0.8,
          });
        } catch {
          // skip failed material meshes
        }
      }),
    );
  }
```

- [ ] **Step 7: Remove dead code**

Remove from `component-browser.ts`:
- `_buildLayerControls()` method (lines 488-511)
- Layer controls HTML in `_updateSidePanel()` — remove the `html += this._buildLayerControls(comp);` line and the layer toggle event listener setup (lines 560-563)
- `_onLayerToggle()` method (lines 668-689)
- `_loadLayer()` method (lines 728-756)
- `_tryLoadMultiMaterial()` method (lines 758-784)
- `_loadFasteners()` method (lines 786-822)
- `_loadWires()` method (lines 824-864)
- Layer group clearing loop in old `loadComponent` (lines 714-717)
- The old `loadComponent` method body (replaced in step 3)

- [ ] **Step 8: Run lint**

Run: `pnpm exec biome check --write viewer/ && pnpm exec tsc --noEmit`
Expected: Clean (or only pre-existing issues).

- [ ] **Step 9: Commit**

```bash
git add viewer/component-browser.ts
git commit -m "refactor: replace component browser layer system with ComponentTree

Drop LAYER_META, _loadLayer, _loadFasteners, _loadWires, and layer
checkboxes. The component browser now fetches a ViewerManifest from
/api/components/{name}/manifest and renders a ComponentTree with the
same toggle/solo/unsolo pattern used by the bot viewer."
```

---

## Task 3: ComponentTree — Handle Single-Component Manifests Gracefully

**Files:**
- Modify: `viewer/component-tree.ts`

The ComponentTree auto-generates assemblies from the kinematic tree. With a single-body, zero-joint manifest the tree should still render a useful hierarchy: the component as root, with fastener and wire groups as children.

- [ ] **Step 1: Verify current behavior with single-body manifest**

Read `_buildAutoAssemblyFromKinematics` in component-tree.ts to understand what happens when there are no joints. The root body will be found, but there will be no sub-assemblies. Verify whether mounts and parts still get rendered as children.

- [ ] **Step 2: If needed, adjust tree building for zero-joint manifests**

If the tree renders only the body with no children, add handling so that the single mount's fastener/wire parts appear. The existing logic in `_buildAutoAssemblyFromKinematics` walks joints to find children — for a zero-joint manifest it should still show mounts and parts on the root body.

Read the method and determine if a fix is needed. If the mount + parts already appear (via `mountsByBody` and `partsByBody` indexing), no code change is required — just document this in a commit message.

- [ ] **Step 3: Run lint + validate**

Run: `make lint && make validate`
Expected: Clean.

- [ ] **Step 4: Commit (if changes were made)**

```bash
git add viewer/component-tree.ts
git commit -m "fix: ComponentTree handles single-body manifests with no joints

Ensures mounts and parts on the root body appear even when there are
no joints (zero sub-assemblies), as happens with component manifests."
```

---

## Task 4: Section Cap Color Migration

**Files:**
- Modify: `viewer/component-browser.ts`

The section cap color callback in loadComponent (Task 2, Step 3) was a placeholder. Refine it to derive colors from manifest materials and node categories.

- [ ] **Step 1: Implement manifest-aware section cap colors**

Update the `setSectionCapColorFn` callback in `loadComponent`:

```typescript
    this.viewport.setSectionCapColorFn((groupName: string) => {
      if (!this._manifest) return null;
      const materials = this._manifest.materials ?? {};

      // Body groups → use component color
      if (groupName.startsWith('body:')) {
        const comp = this.currentComponent;
        if (!comp) return 0xced9e0;
        return tintColor(new THREE.Color(comp.color[0], comp.color[1], comp.color[2]), 0.6);
      }

      // Fastener groups → gold
      if (groupName.startsWith('fastener')) return tintColor(0xd4a843, 0.6);

      // Wire groups → purple
      if (groupName.startsWith('wire') || groupName.startsWith('connector')) return tintColor(0x9179f2, 0.6);

      // Default
      return tintColor(0xced9e0, 0.6);
    });
```

- [ ] **Step 2: Manual test**

Open a component in the browser, enable section plane, verify cap colors:
- Body caps use the component's material color
- Fastener caps are gold
- Wire caps are purple

- [ ] **Step 3: Commit**

```bash
git add viewer/component-browser.ts
git commit -m "fix: section cap colors derived from manifest instead of LAYER_META

Body caps use component color, fasteners get gold, wires get purple —
matching the old LAYER_META behavior but sourced from manifest data."
```

---

## Task 5: Integration Testing + Cleanup

**Files:**
- Modify: `viewer/component-browser.ts` (final cleanup)

- [ ] **Step 1: Manually test representative components**

Open each in the component viewer and verify:
1. **BEC5V** — multi-material body renders, fasteners toggle from tree, section caps colored
2. **STS3215** — servo body renders, wire stubs visible when toggled, tree shows fastener groups
3. **RaspberryPiZero2W** — multi-material body, mounting points visible, markers still work on hover
4. **OV5647** (camera) — body renders, fasteners work

For each, check:
- Tree renders with correct hierarchy
- Toggle visibility works (click visibility dot)
- Solo/unsolo works (right-click, Escape)
- Section plane caps have correct colors
- 3D markers for mounting points and wire ports still highlight on hover
- Steps mode (ShapeScript debugger) still works
- Quick Switcher navigation between components works

- [ ] **Step 2: Remove any remaining dead code**

Check for any remaining references to `LAYER_META`, `ALL_LAYER_IDS`, `layerGroups`, `_onLayerToggle`, `_buildLayerControls`, etc. Remove any found.

- [ ] **Step 3: Run full validation**

Run: `make lint && make validate`
Expected: Clean.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "cleanup: remove remaining LAYER_META references

Final cleanup after component browser→ComponentTree migration.
All layer system code is gone, component browser uses the same
manifest+tree pattern as the bot viewer."
```

---

## Verification Summary

After all tasks complete, the following should be true:

1. `LAYER_META` and `ALL_LAYER_IDS` no longer exist in `component-browser.ts`
2. `_loadLayer`, `_loadFasteners`, `_loadWires`, `_tryLoadMultiMaterial`, `_buildLayerControls` are gone
3. The component viewer fetches `/api/components/{name}/manifest` and builds a `ComponentTree`
4. Toggle, solo, unsolo all work via tree callbacks → DesignScene → Three.js
5. Section cap colors derive from manifest materials, not LAYER_META
6. Side panel specs (dimensions, mass, servo details, mounting points, wire ports) are unchanged
7. Steps mode, section plane, measurements, axis gizmo, quick switcher all work unchanged
8. `make lint && make validate` pass clean
