# Data-Oriented / Interpreter Refactor for botcad

## Summary

Refactor botcad to consistently apply the data-oriented / interpreter pattern already proven by ShapeScript. Every module becomes exactly one of: **Data** (declares what something is), **Transformer** (computes derived data), or **Interpreter** (reads data, produces output). This eliminates ~2000 lines of duplication, unifies the geometry pipeline on ShapeScript IR, and ensures materials/appearances flow from a single source of truth to all consumers.

## Motivation

The codebase is halfway through this transition. ShapeScript is the exemplar — clean IR with separate interpreters. But bracket geometry, mass computation, and color assignment still use imperative code with multiple independent implementations that can drift. Specific problems:

- `bracket.py` (1415 lines) calls build123d directly; `emit_bracket.py` (720 lines) reimplements the same logic to produce ShapeScript IR
- `cad.py:_build_body_solid()` (250 lines) mirrors `emit_body.py:emit_body_ir()` but calls build123d instead of emitting IR
- `cad.py:_compute_world_positions()` reimplements the skeleton's world transform walk
- `_generate_solid()` in `mindsim/server.py` is an ad-hoc interpreter with manual sub-part positioning
- Body colors are computed on-demand by each emitter rather than stored as data
- PLA density and FDM print parameters are hardcoded magic numbers in two places
- Viewer layer colors are hardcoded hex in JavaScript, disconnected from the Python palette

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

- **Data:** `component.py`, `skeleton.py`, `colors.py`, `connectors.py`, `fasteners.py`, `bracket.py` (bracket specs + ShapeScript emission)
- **Transformer:** `packing.py`, `routing.py`, a new `enrich.py` (component dimensions, world transforms)
- **Interpreter:** `emit/cad.py`, `emit/mujoco.py`, `render_svg.py`, `shapescript/backend_occt.py`, `mindsim/server.py`

### Material & Appearance as Data

**Material** declares physical properties that affect mass computation:

```python
@dataclass(frozen=True)
class Material:
    name: str                    # "PLA", "TPU", "aluminum"
    density: float               # kg/m^3 (PLA=1200, TPU=1120, aluminum=2700)
    process: PrintProcess | None # FDM params if 3D-printed

@dataclass(frozen=True)
class PrintProcess:
    wall_layers: int = 2         # perimeter count
    nozzle_width: float = 0.0004 # 0.4mm
    infill: float = 0.20         # 20%

# Standard instances
PLA = Material("PLA", 1200.0, PrintProcess())
TPU = Material("TPU", 1120.0, PrintProcess(infill=0.15))
ALUMINUM = Material("aluminum", 2700.0, None)
```

Fabricated bodies get `material=PLA` (or whatever). Purchased components keep their fixed `.mass`. The mass computation in cad.py reads `body.material` instead of hardcoded constants.

**Appearance** declares visual properties, stored on Body and Component:

```python
@dataclass(frozen=True)
class Appearance:
    color: RGBA                   # primary color
    metallic: float = 0.0         # 0=plastic, 1=metal (for viewer PBR)
    roughness: float = 0.7        # surface roughness
    opacity: float = 1.0          # transparency
```

Every Body and Component gets an `appearance` field. Emitters never compute colors — they read `body.appearance.color` and emit it.

**Color derivation flow:**
1. Component factories set appearance from the semantic palette (`COLOR_POWER_BATTERY` etc.)
2. During `solve()`, fabricated bodies get appearance based on their role (structural -> light gray, etc.) — assigned once, stored in Body
3. Purchased bodies (servos, horns) inherit appearance from their Component
4. All emitters (STEP, MuJoCo, viewer) read `body.appearance` — zero recomputation
5. Viewer API serves appearances as JSON; JS reads them instead of hardcoding hex

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

### Component Assembly Tree

Today ServoSpec declares raw data (shaft_offset, mounting_points). Each consumer independently figures out where the horn goes, where the coupler goes. `_generate_solid()` in `mindsim/server.py` is the worst example — ad-hoc positioning with `.locate()` bugs.

**After:** ServoSpec declares its assembly tree — the full spatial relationship of sub-parts:

```python
@dataclass(frozen=True)
class AssemblyNode:
    """A positioned sub-part in a component assembly."""
    part_id: str               # "horn_front", "coupler", "bracket"
    shapescript: ShapeScript    # geometry IR
    transform: Transform       # position + orientation relative to parent
    appearance: Appearance
    children: tuple[AssemblyNode, ...] = ()

class ServoSpec(Component):
    ...
    def assembly(self, spec: BracketSpec) -> AssemblyNode:
        """Full assembly tree: servo -> horns, bracket, coupler, fasteners."""
```

Single interpreter for all consumers:
- Viewer API: walks the tree, serves each node as a positioned STL layer
- CAD emitter: walks the tree, executes ShapeScript for each node
- Render pipeline: walks the tree for assembly instruction renders

`_generate_solid()` in `mindsim/server.py` dies completely — replaced by `servo.assembly(spec)` + a generic tree walker.

### Emitter Cleanup — Deduplicate

| Duplication | Fix |
|------------|-----|
| `cad.py:_compute_world_positions()` reimplements skeleton's world transform walk | Delete it, read `body.world_pos` |
| `_body_color_rgb()` computed in both cad.py and mujoco.py | Move to `Body.appearance` |
| `skeleton.py:_compute_component_dimensions()` runs ShapeScript inside the data model | Move to a transformer module (runs after solve, before emit) |
| Viewer JS hardcodes layer colors | Serve from API via `appearance.color` |

### Documentation

Update `CLAUDE.md` to codify the pattern:

```markdown
## Architecture: Data-Oriented / Interpreter Pattern

Every module in botcad is exactly one of:

- **Data** — declares what something is (frozen dataclasses, no side effects)
  - `component.py`, `skeleton.py`, `colors.py`, `connectors.py`, `fasteners.py`, `bracket.py`
- **Transformer** — reads data, computes derived data, writes back
  - `packing.py`, `routing.py`, `enrich.py`
- **Interpreter** — reads data model, produces output artifacts
  - `emit/cad.py`, `emit/mujoco.py`, `render_svg.py`, `shapescript/backend_occt.py`

Rules:
- Data never imports interpreters
- Interpreters never mutate the data model
- Transformers run in a defined order (solve -> enrich -> emit)
- ShapeScript is the IR for all geometry — no direct build123d calls outside backend_occt.py
- Material and Appearance are data on Body/Component — emitters read, never compute
```

## Sequencing

The work decomposes into two phases plus a documentation pass:

**Phase A: Bracket IR + Material/Appearance data types**
- Add `Material`, `PrintProcess`, `Appearance` data types
- Add `appearance` field to Body and Component
- Migrate bracket.py to emit ShapeScript; merge emit_bracket.py into it
- Replace `_build_body_solid()` in cad.py with IR execution
- Replace hardcoded density/infill with `body.material`
- Replace `_body_color_rgb()` with `body.appearance`

**Phase B: Component Assembly Tree + Viewer unification**
- Add `AssemblyNode` and `ServoSpec.assembly()`
- Kill `_generate_solid()` in mindsim/server.py
- Viewer API serves appearance colors from Python
- Remove hardcoded hex from viewer JS
- Move `_compute_component_dimensions()` out of skeleton into transformer

**Phase C: Documentation**
- Update CLAUDE.md with architecture section
- Update module docstrings to declare their role (Data/Transformer/Interpreter)

Phases A and B are sequential (B depends on bracket IR from A). Phase C can happen alongside either.

## Downstream Effects

- **All bots** (so101_arm, wheeler_base, wheeler_arm) will need mesh regeneration after bracket.py changes
- **Test baselines** (shapescript snapshots) will need updating
- **Clearance validation** should be unaffected — it already reads from Body solids
- **The STS3215 component fixes** (horn Z-positioning, coupler positioning, envelope sizing) are subsumed by this refactor — the assembly tree makes those bugs structurally impossible
