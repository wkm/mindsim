# Data-Oriented / Interpreter Refactor for botcad

## Summary

Refactor botcad to consistently apply the data-oriented / interpreter pattern already proven by ShapeScript. Every module becomes exactly one of: **Data** (declares what something is), **Transformer** (computes derived data), or **Interpreter** (reads data, produces output). This eliminates ~1200 lines of duplication, unifies the geometry pipeline on ShapeScript IR, and ensures materials/appearances flow from a single source of truth to all consumers.

## Motivation

The codebase is halfway through this transition. ShapeScript is the exemplar — clean IR with separate interpreters. But bracket geometry, mass computation, and color assignment still use imperative code with multiple independent implementations that can drift. Specific problems:

- `bracket.py` (1415 lines) calls build123d directly; `emit_bracket.py` (720 lines) reimplements the same logic to produce ShapeScript IR
- `cad.py:_build_body_solid()` (250 lines) mirrors `emit_body.py:emit_body_ir()` but calls build123d instead of emitting IR
- `cad.py:_compute_world_positions()` reimplements the skeleton's world transform walk (skeleton already computes and stores `body.world_pos` — this is a straightforward deletion)
- `_generate_solid()` in `mindsim/server.py` is an ad-hoc interpreter with manual sub-part positioning
- Body colors are computed on-demand by each emitter rather than stored as data
- PLA density and FDM print parameters are hardcoded magic numbers in two places
- Viewer layer colors are hardcoded hex in JavaScript, disconnected from the Python palette

**Estimated duplication eliminated:** `emit_bracket.py` (720 lines) + `_build_body_solid()` (~250 lines) + `_generate_solid()` in server.py (~60 lines) + `_compute_world_positions()` (~40 lines) + scattered color/density duplication (~50 lines) = ~1120 lines.

## Design

### Principle

Every module falls into exactly one of three roles:

| Role | What it does | What it never does |
|------|-------------|-------------------|
| **Data** | Declares what something *is* — dimensions, relationships, positions, materials, appearances | Calls build123d, generates meshes, writes files |
| **Transformer** | Reads data, computes derived data, writes it back to the data model | Generates output artifacts |
| **Interpreter** | Reads data model, produces output (STL, XML, SVG, viewer JSON) | Mutates the data model |

The invariant: data flows one direction. Skeleton -> (transformer enriches) -> (interpreters read). No interpreter computes something that should be in the data model. No data object calls an interpreter.

**Module classification after refactor:**

- **Data:** `component.py`, `skeleton.py`, `colors.py`, `materials.py`, `connectors.py`, `fasteners.py`
- **Data + Transformer (pure derivation):** `bracket.py` — `BracketSpec` is data; `bracket_solid()` etc. are pure functions that read specs and produce ShapeScript IR. These are transformer functions co-located with the data they derive from.
- **Transformer:** `packing.py`, `routing.py`, a new `enrich.py` (component dimensions, world transforms)
- **Interpreter:** `emit/cad.py`, `emit/mujoco.py`, `render_svg.py`, `shapescript/backend_occt.py`, `mindsim/server.py`

**Note on co-located transformers:** Pure functions that derive IR from frozen data (e.g., `bracket_solid(servo, spec) -> ShapeScript`) are acceptable on Data modules. They have no side effects, don't import interpreters, and are simple enough to live alongside the specs they transform. The key test: does it mutate anything? Does it produce output artifacts? If no to both, it can stay with the data.

### Material & Appearance as Data

**File location:** New file `botcad/materials.py` for `Material` and `PrintProcess`. `Appearance` goes in `botcad/component.py` alongside the existing Component classes. `colors.py` remains as the palette of color constants — `Appearance` references colors from it.

**Material** declares physical properties that affect mass computation:

```python
# botcad/materials.py

@dataclass(frozen=True)
class PrintProcess:
    wall_layers: int = 2         # perimeter count
    nozzle_width: float = 0.0004 # 0.4mm
    infill: float = 0.20         # 20%

@dataclass(frozen=True)
class Material:
    name: str                    # "PLA", "TPU", "aluminum"
    density: float               # kg/m^3 (PLA=1200, TPU=1120, aluminum=2700)
    process: PrintProcess | None # FDM params if 3D-printed

# Standard instances
PLA = Material("PLA", 1200.0, PrintProcess())
TPU = Material("TPU", 1120.0, PrintProcess(infill=0.15))
ALUMINUM = Material("aluminum", 2700.0, None)
```

Fabricated bodies get `material=PLA` (or whatever). Purchased components keep their fixed `.mass`. The mass computation in cad.py reads `body.material` instead of hardcoded constants.

**Appearance** declares visual properties:

```python
# botcad/component.py

@dataclass(frozen=True)
class Appearance:
    color: RGBA                   # primary color
    metallic: float = 0.0         # 0=plastic, 1=metal (for viewer PBR)
    roughness: float = 0.7        # surface roughness
    opacity: float = 1.0          # transparency
```

**Migration from `Component.color`:** The existing `color: RGBA` field on Component is replaced by `appearance: Appearance`. All component factories (8+ files in `botcad/components/`) are updated from `color=COLOR_FOO.rgba` to `appearance=Appearance(color=COLOR_FOO.rgba)`. This is a mechanical find-and-replace.

**Body fields:** Body is mutable (not frozen) with fields populated at different lifecycle stages. New fields with defaults:

```python
# botcad/skeleton.py Body class
material: Material | None = None        # set at construction for fabricated bodies
appearance: Appearance | None = None    # set during solve() based on body role
```

Fabricated bodies get `material=PLA` at construction time (in design.py files). Appearance is assigned during `solve()` based on role (structural -> light gray, wheel -> dark gray, etc.) and stored on the Body. Purchased bodies get appearance from their Component during `_create_purchased_bodies()`.

### Bracket IR — Kill the Duplication

Today `bracket.py` calls build123d directly, and `emit_bracket.py` reimplements the same logic to produce ShapeScript IR. ~1400 lines duplicated.

**Before:**
```
bracket.py  --build123d-->  Solid       (used by cad.py)
emit_bracket.py ---------->  ShapeScript (used by emit_body.py)
```

**After:**
```
bracket.py  -------------->  ShapeScript (single source)
    |
backend_occt.py ---------->  Solid       (when needed)
```

Concretely:
- `bracket_solid()`, `cradle_solid()`, `coupler_solid()`, `bracket_envelope()`, `cradle_envelope()`, `servo_solid()` all return `ShapeScript` instead of `build123d.Solid`
- `emit_bracket.py` merges into `bracket.py` — no more parallel file
- Any consumer that needs an actual Solid calls `backend_occt.execute(bracket_solid(servo, spec))`
- `cad.py`'s `_build_body_solid()` (250 lines of direct build123d) gets replaced by: build ShapeScript via `emit_body_ir()`, execute via backend_occt
- The `@lru_cache` on bracket functions is replaced by ShapeScript's content-hash caching

**What this kills:**
- `emit_bracket.py` (720 lines) — merged into bracket.py
- `_build_body_solid()` in cad.py (~250 lines) — replaced by IR execution
- The dual code path (ShapeScript vs build123d) in cad.py

**Validation strategy:** Before deleting `_build_body_solid()`, add a one-time validation test that asserts `backend_occt.execute(emit_body_ir(body))` produces volumes within 0.1% of the direct build123d path. Run this across all bodies of all bots (so101_arm, wheeler_base, wheeler_arm) to confirm equivalence. Once confirmed, delete the build123d path and remove the validation test.

### Component Assembly Tree

Today ServoSpec declares raw data (shaft_offset, mounting_points). Each consumer independently figures out where the horn goes, where the coupler goes. `_generate_solid()` in `mindsim/server.py` is the worst example — ad-hoc positioning with `.locate()` bugs.

**After:** A standalone pure function builds the assembly tree from component specs:

```python
# botcad/assembly.py — a Transformer

@dataclass(frozen=True)
class AssemblyNode:
    """A positioned sub-part in a component assembly."""
    part_id: str               # "horn_front", "coupler", "bracket"
    shapescript: ShapeScript    # geometry IR
    transform: Transform       # position + orientation relative to parent
    appearance: Appearance
    children: tuple[AssemblyNode, ...] = ()

def servo_assembly(servo: ServoSpec, spec: BracketSpec) -> AssemblyNode:
    """Build the full assembly tree for a servo component.

    Pure function: reads specs, produces tree. No side effects.
    Returns: servo -> horns, bracket, coupler, fasteners.
    """
```

This is a standalone Transformer function, not a method on ServoSpec. ServoSpec stays pure Data. The assembly tree is computed, not stored — it's a derivation from the specs.

**Single interpreter for all consumers:**
- Viewer API: walks the tree, serves each node as a positioned STL layer
- CAD emitter: walks the tree, executes ShapeScript for each node
- Render pipeline: walks the tree for assembly instruction renders

`_generate_solid()` in `mindsim/server.py` dies completely — replaced by `servo_assembly(servo, spec)` + a generic tree walker.

### Emitter Cleanup — Deduplicate

| Duplication | Fix |
|------------|-----|
| `cad.py:_compute_world_positions()` reimplements skeleton's world transform walk | Delete it — skeleton already stores `body.world_pos`. Update callers to read directly. |
| `_body_color_rgb()` computed in both cad.py and mujoco.py | Move to `Body.appearance` (set once during solve) |
| `skeleton.py:_compute_component_dimensions()` runs ShapeScript inside the data model | Move to `enrich.py` transformer. **Important:** this must run *during* solve (after `_collect_tree`, before `solve_packing`), not after solve. The packing solver depends on resolved component bounding boxes. The transformer pipeline is: collect_tree -> enrich (component dims) -> solve_packing -> enrich (world transforms, appearances) -> emit. |
| Viewer JS hardcodes layer colors | Serve from API via `appearance.color` |

### Documentation

Update `CLAUDE.md` to codify the pattern:

```markdown
## Architecture: Data-Oriented / Interpreter Pattern

Every module in botcad is exactly one of:

- **Data** — declares what something is (frozen dataclasses, no side effects)
  - `component.py`, `skeleton.py`, `colors.py`, `materials.py`, `connectors.py`, `fasteners.py`
  - Pure derivation functions (e.g., `bracket_solid(servo, spec) -> ShapeScript`) may live on
    data modules when they are side-effect-free and don't import interpreters.
- **Transformer** — reads data, computes derived data, writes back
  - `packing.py`, `routing.py`, `enrich.py`, `assembly.py`
- **Interpreter** — reads data model, produces output artifacts
  - `emit/cad.py`, `emit/mujoco.py`, `render_svg.py`, `shapescript/backend_occt.py`

Rules:
- Data never imports interpreters
- Interpreters never mutate the data model
- Transformers run in a defined order:
  collect_tree -> enrich(component dims) -> solve_packing -> enrich(world transforms, appearances) -> emit
- ShapeScript is the IR for all geometry — no direct build123d calls outside backend_occt.py
- Material and Appearance are data on Body/Component — emitters read, never compute
```

## Sequencing

The work decomposes into two phases plus a documentation pass:

**Phase A: Bracket IR + Material/Appearance data types**
- Add `Material`, `PrintProcess` in new `botcad/materials.py`
- Add `Appearance` in `botcad/component.py`, migrate `Component.color` -> `Component.appearance`
- Add `material` and `appearance` fields to Body
- Migrate bracket.py to emit ShapeScript; merge emit_bracket.py into it
- Replace `_build_body_solid()` in cad.py with IR execution (validate volume equivalence first)
- Replace hardcoded density/infill with `body.material`
- Replace `_body_color_rgb()` with `body.appearance`

**Phase B: Component Assembly Tree + Viewer unification**
- Add `AssemblyNode` and `servo_assembly()` in new `botcad/assembly.py`
- Kill `_generate_solid()` in mindsim/server.py
- Viewer API serves appearance colors from Python
- Remove hardcoded hex from viewer JS
- Move `_compute_component_dimensions()` out of skeleton into `enrich.py`

**Phase C: Documentation**
- Update CLAUDE.md with architecture section
- Update module docstrings to declare their role (Data/Transformer/Interpreter)

Phases A and B are sequential (B depends on bracket IR from A). Phase C can happen alongside either.

## Downstream Effects

- **All bots** (so101_arm, wheeler_base, wheeler_arm) will need mesh regeneration after bracket.py changes
- **Test baselines** (shapescript snapshots) will need updating
- **Clearance validation** should be unaffected — it already reads from Body solids
- **The STS3215 component fixes** (horn Z-positioning, coupler positioning, envelope sizing) are subsumed by this refactor — the assembly tree makes those bugs structurally impossible
