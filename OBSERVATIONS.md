# Botcad Observations & Improvement Backlog

Captured 2026-03-17 from a deep architecture review of `botcad/`.
Check items off as they're resolved. Add new findings at the bottom.

---

## High Severity

- [ ] **H1. Mass/inertia computed twice with different models**
  `packing.py:_compute_mass_inertia` (thin-walled box heuristic) computes mass, COM,
  and inertia during `solve()`. Then `emit/cad.py:_update_mass_from_solid` recomputes
  and **overwrites** using actual CAD geometry (FDM perimeter + 20% infill). Code
  between `solve()` and `emit()` sees stale values. The parallel axis theorem code is
  nearly duplicated line-for-line between the two files. Resolution: either defer mass
  computation to `emit()`, or run the CAD-based model in `solve()` and remove the
  heuristic.

- [ ] **H2. `_as_solid()` duplicated with different robustness**
  `bracket.py:97-128` has a comprehensive version (handles ShapeList, multi-solid
  Compounds). `cad.py:545-561` has a simpler version that silently passes through
  malformed results. Consolidate into one shared implementation.

- [ ] **H3. Type mismatch: `_wireport_local` receives `str` instead of `BusType`**
  `routing.py:329` passes `"uart_half_duplex"`, `routing.py:445` passes `"csi"`.
  The function signature expects `BusType` and uses `is` comparison (line 156).
  Works by accident via StrEnum interning. Fix: pass `BusType.UART_HALF_DUPLEX` /
  `BusType.CSI` and change `is` to `==`.

- [ ] **H4. Bot monkey-patched during `emit()`**
  `emit/cad.py:478-479` sets `bot._assembly_parts` and `bot._part_modules` —
  undeclared attributes on the Bot dataclass. `emit_cad_for_module()` reads them
  later. Implicit coupling invisible to the type system. Fix: return these as
  explicit outputs or add declared fields.

---

## Medium Severity

- [ ] **M1. Three independent Z-to-axis rotation implementations**
  `mujoco.py:_z_to_axis_quat` (returns Quat|None), `component_renders.py:_axis_quat`
  (returns str|None), `geometry.py:rotation_between` (general quaternion rotation).
  All compute a quaternion rotating Z to a target axis. Consolidate on
  `rotation_between()` from `geometry.py`.

- [ ] **M2. `_body_color_rgb` defined in mujoco.py, imported by cad.py**
  `cad.py` does `from botcad.emit.mujoco import _body_color_rgb`. Color logic should
  live in `colors.py` which already exists as the centralized color module.

- [ ] **M3. `_parse_axis` silently defaults to Z for unknown strings**
  `skeleton.py:527-539` — `mapping.get(axis.lower(), (0, 0, 1))` silently defaults.
  A typo like `"vertical"` becomes Z with no warning. Should raise `ValueError`.

- [ ] **M4. Mutable dataclass mixes user input + solver output**
  `Mount`, `Joint`, `Body` all have user-set fields and solver-output fields
  (`resolved_pos`, `solved_servo_center`, `solved_dimensions`, etc.) on the same
  object. No clear separation between "what the user declared" and "what the solver
  computed". Consider separating into input vs solved types, or at least documenting
  the boundary clearly.

- [ ] **M5. `_build_parent_map` rebuilt twice per `solve_routing()`**
  Called once by `_route_servo_bus` (line 333) and once by `_route_camera_csi`
  (line 460). Should build once and pass through.

- [ ] **M6. `emit()` hardcodes sequential emitter chain**
  `skeleton.py:467-524` — 60 lines of sequential `from X import Y; Y(self, dir)`.
  A registration pattern would allow conditional emitters (skip renders in CI).

- [ ] **M7. Edge detection duplicated in assembly_renders.py**
  `render3d.py:_detect_edges` exists as a shared function, but
  `assembly_renders.py:247-261` reimplements the same 4-neighbor segmentation
  boundary detection inline.

---

## Low Severity

- [ ] **L1. `renders.py` does manual grid compositing**
  `renders.py:140-312` manually computes grid layout instead of using
  `composite.py:grid()` which was created specifically to replace it.

- [ ] **L2. Wire radius dicts duplicated across emitters**
  `cad.py:357-361` and `mujoco.py:612-616` both define wire radius maps per BusType.
  Values are intentionally different (channel > wire) but should be co-located,
  perhaps in `component.py` or `colors.py` alongside other shared constants.

- [ ] **L3. No validation that `Bot.root` is set before `solve()`**
  `skeleton.py:432` — if root is None, `all_bodies` and `all_joints` stay empty and
  solve silently succeeds with no output. Should raise early.

- [ ] **L4. `Renderer3D` calls `update_scene` 3x per frame**
  `render3d.py:420-443` — once per render pass (color, segmentation, depth). May be
  required by MuJoCo's API, but worth investigating whether scene can be cached
  across passes.

- [ ] **L5. `_component_category` still instantiates every factory on first call**
  The registry is cached after first build, but the build itself calls every callable
  in `botcad.components.*` as `obj()`. A static mapping dict would be simpler and
  wouldn't execute factory code.

---

## Test Gaps

- [ ] **T1. No unit tests for wire routing** — `solve_routing` is completely untested.
- [ ] **T2. Packing overlap tests xfail** for 2 of 3 bots — known issue with no resolution.
- [ ] **T3. No bracket geometry tests** — `bracket_solid`, `coupler_solid` correctness
  untested (volume, bounding box, hole placement).
- [ ] **T4. No mesh quality tests** — STL manifoldness, vertex counts, watertightness.
- [ ] **T5. Shared test fixtures missing** — bot loading duplicated across test files;
  no conftest fixtures for common patterns.
- [ ] **T6. No tests for joint property validation** — torque, speed, range consistency
  between ServoSpec and Joint declarations.
