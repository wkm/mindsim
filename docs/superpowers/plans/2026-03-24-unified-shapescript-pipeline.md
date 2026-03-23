# Unified ShapeScript Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ShapeScript the single geometry path. All consumers (viewer, MuJoCo XML, STEP export, renders) read from the same IR. Eliminate `bot.emit()`, pre-generated STLs, and all direct build123d usage outside `OcctBackend`.

**Architecture:** Bot IR (skeleton + solved ShapeScript programs) is the single source of truth. `build_cad()` executes ShapeScript programs via `OcctBackend` and caches results in `DiskCache`. All geometry consumers — viewer mesh server, MuJoCo XML emitter, STEP exporter — read from this cached IR. No consumer generates geometry independently. `PrebuiltOp` is eliminated by adding two new ShapeScript ops.

**Tech Stack:** ShapeScript IR, OcctBackend (build123d), DiskCache (shelve-based), FastAPI server

**Ref:** `docs/superpowers/specs/2026-03-22-data-oriented-refactor-design.md`

---

## Overview

Seven tasks, roughly in dependency order:

1. **New ShapeScript ops** — `RegularPolygonExtrudeOp` + `ChamferByFaceOp` → kill `PrebuiltOp`
2. **Delete direct build123d component factories** — wire `make_component_solid()` through ShapeScript
3. **Delete `_build_body_solid()` and `SHAPESCRIPT=0` fallback** — `emit_body_ir()` becomes the only path
4. **Unify server.py onto ShapeScript caching** — remove `_solid_cache`/`_stl_cache`, serve from `DiskCache`
5. **On-demand MuJoCo XML generation** — `emit_mujoco()` reads from cached ShapeScript results, no pre-generated files needed
6. **Delete `bot.emit()` as monolithic step** — replace with lazy per-consumer generation
7. **Cleanup** — remove dead code, unused imports, update tests

## File Map

**New files:**
- None (all changes are to existing files)

**Modified files (by task):**

| Task | Files | Change |
|------|-------|--------|
| 1 | `botcad/shapescript/ops.py` | Add `RegularPolygonExtrudeOp`, `ChamferByFaceOp` |
| 1 | `botcad/shapescript/backend_occt.py` | Handle new ops in executor |
| 1 | `botcad/shapescript/program.py` | Add convenience methods |
| 1 | `botcad/shapescript/emit_components.py` | Rewrite `fastener_script()` without `PrebuiltOp` |
| 1 | `tests/test_shapescript_ops.py` | Tests for new ops |
| 2 | `botcad/emit/cad.py` | Rewrite `make_component_solid()` to execute ShapeScript |
| 2 | `botcad/components/camera.py` | Delete `camera_solid()` |
| 2 | `botcad/components/battery.py` | Delete `battery_solid()` |
| 2 | `botcad/components/compute.py` | Delete `raspberry_pi_zero_solid()` |
| 2 | `botcad/connectors.py` | Delete `connector_solid()`, `receptacle_solid()` |
| 2 | `botcad/fasteners.py` | Delete `fastener_solid()` |
| 2 | `botcad/emit/cad.py` | Delete `_horn_solid()`, `_make_wheel_solid()`, `_make_bearing_solid()`, `_wire_segment_solid()` |
| 3 | `botcad/emit/cad.py` | Delete `_build_body_solid()` (~290 lines), `_cut_camera_features()`, `_child_outer_envelope_local()`, `_child_clearance_volume()`, `_wire_channel()` |
| 3 | `botcad/emit/cad.py` | Remove `SHAPESCRIPT=0` env var check in `build_cad()` |
| 4 | `mindsim/server.py` | Remove `_solid_cache`, `_stl_cache`; serve STLs from DiskCache |
| 4 | `mindsim/server.py` | Rewrite `_generate_solid()` / `_generate_stl_bytes()` to use ShapeScript execution |
| 5 | `botcad/emit/mujoco.py` | Accept pre-computed body_solids dict (from cached ShapeScript) |
| 5 | `mindsim/server.py` or `main.py` | On-demand MuJoCo XML generation endpoint or lazy builder |
| 6 | `botcad/skeleton.py` | Remove or restructure `emit()` method |
| 6 | `botcad/emit/cad.py` | Restructure `emit_cad()` into lazy per-consumer functions |
| 7 | Various | Delete dead imports, unused functions, update test expectations |

---

### Task 1: New ShapeScript Ops — Kill PrebuiltOp

**Files:**
- Modify: `botcad/shapescript/ops.py`
- Modify: `botcad/shapescript/backend_occt.py`
- Modify: `botcad/shapescript/program.py`
- Modify: `botcad/shapescript/emit_components.py:435-513` (fastener_script + _build_fastener_head)
- Create: `tests/test_shapescript_new_ops.py`

**Context:** `PrebuiltOp` injects raw build123d solids into the IR, making programs non-serializable and breaking caching. Two ops are needed: `RegularPolygonExtrudeOp` (hex socket recess) and `ChamferByFaceOp` (top-edge chamfer on cylinder). The Phillips head case already uses only `BoxOp` + `CutOp`.

- [ ] **Step 1: Write failing test for RegularPolygonExtrudeOp**

```python
# tests/test_shapescript_new_ops.py
def test_regular_polygon_extrude_creates_solid():
    """A 6-sided polygon extruded to 5mm should produce a hex prism."""
    prog = ShapeScript()
    hex = prog.regular_polygon_extrude(radius=0.003, sides=6, height=0.005)
    prog.output_ref = hex
    result = OcctBackend().execute(prog)
    solid = result.shapes[hex.id]
    assert solid.volume > 0
    # Hex prism volume = (3√3/2) * r² * h ≈ 1.17e-7 m³
    import math
    expected = (3 * math.sqrt(3) / 2) * 0.003**2 * 0.005
    assert abs(solid.volume - expected) / expected < 0.05
```

- [ ] **Step 2: Run test — expect FAIL (method doesn't exist)**

Run: `uv run pytest tests/test_shapescript_new_ops.py::test_regular_polygon_extrude_creates_solid -v`

- [ ] **Step 3: Add RegularPolygonExtrudeOp to ops.py**

```python
@dataclass(frozen=True)
class RegularPolygonExtrudeOp:
    """Extrude a regular N-sided polygon to a given height."""
    ref: ShapeRef
    radius: float
    sides: int
    height: float
    align: Align3 = ALIGN_CENTER
    tag: str | None = None
```

Add to `Op` union type.

- [ ] **Step 4: Handle in backend_occt.py executor**

```python
case RegularPolygonExtrudeOp(ref=ref, radius=r, sides=n, height=h, align=align, tag=tag):
    from build123d import RegularPolygon, extrude
    profile = RegularPolygon(r, n)
    solid = extrude(profile, h)
    solid = _apply_align(solid, align)  # reuse existing align logic
    shapes[ref.id] = solid
    if tag:
        tags.register(ref, tag)
```

- [ ] **Step 5: Add convenience method to program.py**

```python
def regular_polygon_extrude(self, radius: float, sides: int, height: float,
                            align: Align3 = ALIGN_CENTER, tag: str | None = None) -> ShapeRef:
    ref = self._next_ref()
    self.ops.append(RegularPolygonExtrudeOp(ref, radius, sides, height, align, tag))
    return ref
```

- [ ] **Step 6: Run test — expect PASS**

- [ ] **Step 7: Write failing test for ChamferByFaceOp**

```python
def test_chamfer_by_face_reduces_volume():
    """Chamfering top face edges of a cylinder should reduce volume."""
    prog = ShapeScript()
    cyl = prog.cylinder(0.003, 0.005)
    original_vol_prog = ShapeScript()
    orig = original_vol_prog.cylinder(0.003, 0.005)
    original_vol_prog.output_ref = orig

    chamfered = prog.chamfer_by_face(cyl, axis="z", end="max", size=0.0003)
    prog.output_ref = chamfered

    backend = OcctBackend()
    orig_result = backend.execute(original_vol_prog)
    cham_result = backend.execute(prog)

    assert cham_result.shapes[chamfered.id].volume < orig_result.shapes[orig.id].volume
```

- [ ] **Step 8: Add ChamferByFaceOp to ops.py**

```python
@dataclass(frozen=True)
class ChamferByFaceOp:
    """Chamfer all edges of the face at the max/min extent along an axis."""
    ref: ShapeRef
    target: ShapeRef
    axis: str  # "x", "y", or "z"
    end: str   # "max" or "min"
    size: float
```

- [ ] **Step 9: Handle in backend_occt.py**

```python
case ChamferByFaceOp(ref=ref, target=t, axis=axis, end=end, size=size):
    from build123d import Axis, chamfer
    s = shapes[t.id]
    axis_obj = {"x": Axis.X, "y": Axis.Y, "z": Axis.Z}[axis]
    faces = s.faces().sort_by(axis_obj)
    face = faces[-1] if end == "max" else faces[0]
    try:
        shapes[ref.id] = chamfer(face.edges(), size)
    except Exception:
        shapes[ref.id] = s  # chamfer failed, pass through
    tags.propagate_transform(ref, t)
```

- [ ] **Step 10: Add convenience method to program.py**

- [ ] **Step 11: Run both tests — expect PASS**

- [ ] **Step 12: Rewrite fastener_script() without PrebuiltOp**

Replace `_build_fastener_head()` + `prog.prebuilt(head_solid)` in `emit_components.py:435-513` with:

```python
def fastener_script(spec: FastenerSpec, length: float) -> ShapeScript:
    prog = ShapeScript()
    head_r = spec.head_diameter / 2
    head_h = spec.head_height
    shank_r = spec.thread_diameter / 2

    # Head cylinder, aligned so top face is at Z=0
    head = prog.cylinder(head_r, head_h, align=(Align.CENTER, Align.CENTER, Align.MAX))
    # Chamfer top edge
    chamfer_size = min(0.2 * head_h, 0.0003)
    head = prog.chamfer_by_face(head, axis="z", end="max", size=chamfer_size)

    if spec.head_type == HeadType.SOCKET_HEAD_CAP and spec.socket_size > 0:
        hex_r = spec.socket_size / 2 / math.cos(math.radians(30))
        recess_depth = head_h * 0.6
        hex = prog.regular_polygon_extrude(hex_r, 6, recess_depth)
        hex = prog.locate(hex, pos=(0, 0, -recess_depth))
        head = prog.cut(head, hex)
    elif spec.head_type == HeadType.PAN_HEAD_PHILLIPS:
        # Cross recess — two intersecting boxes
        slot_w = spec.head_diameter * 0.12
        slot_l = spec.head_diameter * 0.7
        slot_d = head_h * 0.4
        slot1 = prog.box(slot_l, slot_w, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX))
        slot2 = prog.box(slot_w, slot_l, slot_d, align=(Align.CENTER, Align.CENTER, Align.MAX))
        head = prog.cut(head, slot1)
        head = prog.cut(head, slot2)

    # Shank
    shank = prog.cylinder(shank_r, length, align=(Align.CENTER, Align.CENTER, Align.MAX))
    shank = prog.locate(shank, pos=(0, 0, -head_h))

    fastener = prog.fuse(head, shank)
    prog.output_ref = fastener
    return prog
```

- [ ] **Step 13: Delete `_build_fastener_head()` and `PrebuiltOp` import from emit_components.py**

- [ ] **Step 14: Run existing fastener tests + new op tests**

Run: `uv run pytest tests/test_shapescript_new_ops.py tests/test_shapescript_components.py -v`

- [ ] **Step 15: Verify PrebuiltOp is no longer used anywhere**

Run: `grep -r "PrebuiltOp\|prebuilt" botcad/ --include="*.py" -l`
Expected: only `ops.py` (definition) and `backend_occt.py` (handler) — no emitters.

- [ ] **Step 16: Commit**

```bash
git add -A
git commit -m "feat: add RegularPolygonExtrudeOp + ChamferByFaceOp, eliminate PrebuiltOp from fasteners"
```

---

### Task 2: Delete Direct build123d Component Factories

**Files:**
- Modify: `botcad/emit/cad.py:232-289` (`make_component_solid`)
- Modify: `botcad/emit/cad.py` (delete `_horn_solid`, `_make_wheel_solid`, `_make_bearing_solid`, `_wire_segment_solid`, `screw_solid`)
- Modify: `botcad/components/camera.py` (delete `camera_solid`)
- Modify: `botcad/components/battery.py` (delete `battery_solid`)
- Modify: `botcad/components/compute.py` (delete `raspberry_pi_zero_solid`)
- Modify: `botcad/connectors.py` (delete `connector_solid`, `receptacle_solid`)
- Modify: `botcad/fasteners.py` (delete `fastener_solid`)
- Modify: `tests/test_domain_model.py`, `tests/test_cad_emit.py`

**Context:** All these functions have ShapeScript equivalents in `emit_components.py`. The `make_component_solid()` dispatcher in cad.py should execute ShapeScript programs instead of calling direct build123d factories. The `*_solid_solid()` wrappers in bracket.py already do this pattern via `_exec_ir()`.

- [ ] **Step 1: Create `execute_component_script()` helper in cad.py**

A function that takes a component, looks up its `*_script()` emitter, executes the ShapeScript via OcctBackend, and returns the build123d Solid. This replaces all the direct factory dispatch.

```python
def execute_component_script(component: Component) -> Solid | None:
    """Execute the ShapeScript program for a component and return the solid."""
    from botcad.shapescript.emit_components import (
        battery_script, bearing_script, camera_script,
        compute_script, wheel_component_script,
    )
    from botcad.component import BatterySpec, BearingSpec, CameraSpec, ServoSpec

    if isinstance(component, CameraSpec):
        prog = camera_script(component)
    elif isinstance(component, BatterySpec):
        prog = battery_script(component)
    elif isinstance(component, BearingSpec):
        prog = bearing_script(component)
    elif isinstance(component, ServoSpec):
        from botcad.shapescript.emit_servo import servo_script
        prog = servo_script(component)
    elif component.is_wheel:
        prog = wheel_component_script(component)
    elif component.name == "RaspberryPiZero2W":
        prog = compute_script(component)
    elif component.name == "TestFastenerPrism":
        # test component — emit inline or keep as special case
        ...
    else:
        return None

    from botcad.shapescript.backend_occt import OcctBackend
    result = OcctBackend().execute(prog)
    return result.shapes.get(prog.output_ref.id)
```

- [ ] **Step 2: Replace `make_component_solid()` body with `execute_component_script()`**

Keep the `@lru_cache` decorator. The function signature stays the same.

- [ ] **Step 3: Run tests to verify components still build correctly**

Run: `uv run pytest tests/test_cad_emit.py tests/test_domain_model.py -v`

- [ ] **Step 4: Delete direct factory functions**

Delete from each file:
- `camera_solid()` from `botcad/components/camera.py`
- `battery_solid()` from `botcad/components/battery.py`
- `raspberry_pi_zero_solid()` from `botcad/components/compute.py`
- `connector_solid()`, `receptacle_solid()` from `botcad/connectors.py`
- `fastener_solid()` from `botcad/fasteners.py`
- `_horn_solid()`, `_make_wheel_solid()`, `_make_bearing_solid()`, `_wire_segment_solid()`, `screw_solid()` from `botcad/emit/cad.py`

- [ ] **Step 5: Fix all import references to deleted functions**

Run: `grep -rn "camera_solid\|battery_solid\|raspberry_pi_zero_solid\|connector_solid\|receptacle_solid\|fastener_solid\|_horn_solid\|_make_wheel_solid\|_make_bearing_solid\|_wire_segment_solid\|screw_solid" botcad/ mindsim/ tests/ --include="*.py"`

Update each caller to use the ShapeScript path.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor: delete direct build123d component factories, route all through ShapeScript"
```

---

### Task 3: Delete `_build_body_solid()` and SHAPESCRIPT=0 Fallback

**Files:**
- Modify: `botcad/emit/cad.py` (delete ~290 lines of `_build_body_solid` + helpers)
- Modify: `botcad/emit/cad.py:build_cad()` (remove env var check, always use ShapeScript)

**Context:** `_build_body_solid()` is the legacy direct-build123d path for fabricated bodies. `emit_body_ir()` in `botcad/shapescript/emit_body.py` does the same thing via ShapeScript. The `SHAPESCRIPT=0` env var toggle preserved both paths during migration. With the migration complete, delete the legacy path.

- [ ] **Step 1: Verify emit_body_ir covers all cases**

Run existing tests with SHAPESCRIPT=1 (default): `uv run pytest tests/test_cad_emit.py -v`
All should pass — this confirms emit_body_ir handles everything.

- [ ] **Step 2: Remove SHAPESCRIPT=0 fallback from build_cad()**

Find the env var check in `build_cad()` and delete the branch that calls `_build_body_solid()`.

- [ ] **Step 3: Delete `_build_body_solid()` and its helper functions**

Delete from `botcad/emit/cad.py`:
- `_build_body_solid()` (~1031-1320)
- `_cut_camera_features()` (~974-1028)
- `_child_outer_envelope_local()` (~1380-1430)
- `_child_clearance_volume()` (~1454-1504)
- `_wire_channel()` (~1330-1377)
- `_rotate_solid()` (~1443-1451)
- `_orient_z_to_axis()` (~1675-1698) — check if still used by horn STL export first

- [ ] **Step 4: Clean up unused imports**

Run: `uv run ruff check --fix botcad/emit/cad.py`

- [ ] **Step 5: Run full test suite**

Run: `make lint && uv run pytest tests/ -v --tb=short`

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: delete _build_body_solid and SHAPESCRIPT=0 fallback — emit_body_ir is the only path"
```

---

### Task 4: Unify Server onto ShapeScript Caching

**Files:**
- Modify: `mindsim/server.py`
- Modify: `botcad/shapescript/cache.py` (if DiskCache API needs extension)

**Context:** server.py currently maintains its own `_solid_cache` and `_stl_cache` dicts. These should be replaced by reading from the ShapeScript `DiskCache`. When the server needs a component STL, it should: get/build the ShapeScript program → execute via OcctBackend (with DiskCache) → export to STL. The DiskCache handles eviction and invalidation.

- [ ] **Step 1: Audit server.py's _generate_solid and _generate_stl_bytes**

Read `mindsim/server.py` functions that generate geometry. Understand the current cache key scheme.

- [ ] **Step 2: Replace _generate_solid with ShapeScript execution**

Rewrite to:
1. Look up the component's `*_script()` emitter
2. Execute via `OcctBackend` (which hits DiskCache)
3. Return the solid from the execution result

Remove `_solid_cache` and `_stl_cache` dicts.

- [ ] **Step 3: Replace _generate_stl_bytes with ShapeScript STL export**

Use `ExportSTLOp` in the ShapeScript program, or export from the cached solid.

- [ ] **Step 4: Update bot loading to use ShapeScript cache**

Ensure `_load_bot()` in server.py calls `build_cad()` which uses DiskCache, rather than maintaining a separate cache.

- [ ] **Step 5: Run server tests**

Run: `uv run pytest tests/test_viewer_api.py -v`

- [ ] **Step 6: Manual smoke test**

Start the viewer: `uv run mjpython main.py web --bot so101_arm`
Verify components load correctly in the component browser.

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor: unify server.py onto ShapeScript DiskCache, remove duplicate caches"
```

---

### Task 5: On-demand MuJoCo XML Generation

**Files:**
- Modify: `botcad/emit/mujoco.py`
- Modify: `mindsim/server.py` or `main.py`
- Modify: `botcad/skeleton.py`

**Context:** Currently `emit_mujoco()` is called as part of `bot.emit()` which writes bot.xml to disk. Instead, the MuJoCo XML should be generated on-demand from the solved bot state + cached ShapeScript results. The viewer and sim can request it when needed. STL mesh files should also be served on-demand rather than pre-written.

- [ ] **Step 1: Extract `generate_mujoco_xml()` as a pure function**

`emit_mujoco()` currently takes a bot + output_dir and writes files. Refactor to a function that returns the XML string without writing to disk. Keep the file-writing version as a thin wrapper.

- [ ] **Step 2: Add API endpoint for on-demand bot.xml**

```python
@app.get("/api/bots/{bot_name}/bot.xml")
def get_bot_xml(bot_name: str):
    bot, cad_model = _load_bot(bot_name)
    xml_str = generate_mujoco_xml(bot, cad_model)
    return Response(content=xml_str, media_type="application/xml")
```

- [ ] **Step 3: Add API endpoint for on-demand mesh STLs**

```python
@app.get("/api/bots/{bot_name}/meshes/{mesh_name}")
def get_bot_mesh(bot_name: str, mesh_name: str):
    bot, cad_model = _load_bot(bot_name)
    # Look up solid from cad_model.body_solids
    solid = cad_model.body_solids.get(mesh_name.replace(".stl", ""))
    stl_bytes = _solid_to_stl_bytes(solid)
    return Response(content=stl_bytes, media_type="application/octet-stream")
```

- [ ] **Step 4: Update viewer to load from API instead of static files**

The viewer currently fetches `../bots/{bot}/meshes/{name}.stl` as static files. Update to fetch from `/api/bots/{bot}/meshes/{name}.stl`.

- [ ] **Step 5: Run viewer smoke test**

Start viewer, verify bot loads correctly from on-demand endpoints.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: on-demand MuJoCo XML + mesh serving, no pre-generated files needed for viewer"
```

---

### Task 6: Restructure bot.emit()

**Files:**
- Modify: `botcad/skeleton.py` (`Bot.emit()` method)
- Modify: `botcad/emit/cad.py` (`emit_cad()`)
- Modify: `bots/*/design.py` (all bot design files)
- Modify: `scripts/regen_test_renders.py`

**Context:** `bot.emit()` currently generates everything in one shot. With on-demand serving, the viewer no longer needs pre-generated files. However, manufacturing outputs (STEP files, BOMs, assembly guides) and simulation (bot.xml for `mjpython` CLI usage) still need a way to be generated. Restructure so that `emit()` is replaced by explicit per-format generators that can be called independently.

- [ ] **Step 1: Split emit() into per-format methods**

```python
class Bot:
    def write_mujoco(self, output_dir: Path) -> None:
        """Write bot.xml + STL meshes for MuJoCo simulation."""

    def write_step(self, output_dir: Path) -> None:
        """Write STEP assembly files for manufacturing."""

    def write_docs(self, output_dir: Path) -> None:
        """Write BOM, assembly guide, drawings."""

    def write_renders(self, output_dir: Path) -> None:
        """Write overview/sweep render PNGs."""
```

- [ ] **Step 2: Update design.py files to use new API**

Replace `bot.emit(output_dir)` with explicit calls: `bot.write_mujoco(...)`, etc.

- [ ] **Step 3: Update regen_test_renders.py**

- [ ] **Step 4: Keep backward compat temporarily**

`emit()` can call all four methods for now. Mark as deprecated.

- [ ] **Step 5: Run full test suite + regen**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor: split bot.emit() into per-format generators (write_mujoco, write_step, write_docs, write_renders)"
```

---

### Task 7: Cleanup — Dead Code, Unused Imports, Test Updates

**Files:**
- Modify: `botcad/shapescript/ops.py` (remove PrebuiltOp class if no longer referenced)
- Modify: `botcad/shapescript/backend_occt.py` (remove PrebuiltOp handler)
- Modify: `botcad/cad_utils.py` (assess if `as_solid()` is still needed)
- Modify: Various test files
- Delete: `botcad/components/test_fastener.py:test_fastener_solid()` if replaced

- [ ] **Step 1: Remove PrebuiltOp from ops.py and backend**

Verify no remaining references: `grep -r PrebuiltOp botcad/ --include="*.py"`

- [ ] **Step 2: Audit and remove as_solid() if unused**

`as_solid()` in `cad_utils.py` is a build123d type-coercion helper. If all geometry flows through OcctBackend, it may no longer be needed at the call sites.

- [ ] **Step 3: Run ruff to find unused imports**

Run: `uv run ruff check --fix .`

- [ ] **Step 4: Update snapshot baselines if geometry changed**

Run: `uv run pytest tests/test_shapescript_snapshots.py --snapshot-update`

- [ ] **Step 5: Final full validation**

Run: `make validate`

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: cleanup — remove PrebuiltOp, dead code, update snapshots"
```

---

## Dependency Graph

```
Task 1 (new ops)
  └── Task 2 (delete direct factories) — needs new ops for fastener
        └── Task 3 (delete _build_body_solid) — needs Task 2 done first
              └── Task 4 (unify server caching) — needs single geometry path
                    └── Task 5 (on-demand MuJoCo) — needs unified server
                          └── Task 6 (restructure emit) — needs on-demand serving
                                └── Task 7 (cleanup) — final pass
```

Tasks 1-3 are the core migration (ShapeScript-only geometry).
Tasks 4-6 are the architecture change (on-demand serving, no pre-generated files).
Task 7 is cleanup.

## Testing Strategy

- **After each task:** `make lint && uv run pytest tests/ -v`
- **After Task 3:** Verify all three bots build: `uv run pytest tests/test_cad_emit.py::TestAllBotBodySolids -v`
- **After Task 5:** Manual viewer smoke test with `uv run mjpython main.py web --bot so101_arm`
- **After Task 7:** Full validation: `make validate`

## Risk Mitigation

- **Geometry regression:** The ShapeScript equivalents have been tested alongside the direct versions. Volume comparisons in existing tests catch regressions.
- **Performance:** DiskCache avoids re-executing expensive OCCT booleans. First build of a new bot design is slower; subsequent loads are cache hits.
- **Fastener chamfer:** `ChamferByFaceOp` uses OCCT face sorting which can be fragile. The backend has a try/except fallback (pass-through on failure) matching the pattern used by existing fillet/chamfer ops.
