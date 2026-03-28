# DFM Validation & Assembly Sequence System

**Date:** 2026-03-28
**Goal:** Build a design-for-manufacturing validation system that catches physical build issues (fastener access, wire routing, component retention) before printing, with interactive visualization in the web viewer.
**Deadline:** Next weekend (2026-04-05) — wheeler_base printable and sim-validated.

## Context

The wheeler_base CAD pipeline produces meshes, MuJoCo XML, BOM, and assembly guides. All geometry is generated and recent. However, loading the STL into Bambu Studio and visually inspecting the model reveals several build-blocking issues:

- Servo bracket fasteners are inaccessible (no screwdriver clearance)
- Wire routing tubes don't fit connector heads, only bare wire
- Wire tube bends are too tight to push cables through
- Battery has no retention mechanism (falls out)
- Long thin channels without fillets trap support material

These are all detectable from the existing geometry and component specs. The system should catch them automatically and present findings interactively in the viewer.

## Design

### 1. Assembly Sequence Model

The assembly sequence is the foundation. DFM checks run per-step against the geometry state at that point in the build, not against the final assembled state. A fastener that looks blocked in the final state may be accessible at step 3 before the Pi is mounted.

**Replaces the prose assembly guide.** The structured sequence is the source of truth; rendering to human language is trivial later.

#### Units and Dimension Types

All dimensions are SI. To prevent unit confusion, dimension fields use `NewType` wrappers rather than bare floats:

```python
Meters = NewType("Meters", float)
Radians = NewType("Radians", float)
```

These are zero-cost at runtime but caught by type checkers. Display tables in this spec show millimeters for readability; code always stores `Meters`.

For spatial positions and directions, use `Vec3` (already exists in the codebase) which is implicitly in meters.

#### Typed References — the StringId Pattern

Entity names throughout the skeleton (`Body.name`, `Joint.name`, `Mount.label`, `WireRoute.label`) are currently bare strings. This spec introduces a `StringId` pattern: frozen, hashable wrappers that are the canonical way to reference named entities.

```python
@dataclass(frozen=True)
class BodyId:
    name: str

@dataclass(frozen=True)
class JointId:
    name: str

@dataclass(frozen=True)
class ComponentRef:
    body: BodyId
    mount_label: str  # matches Mount.label within body

@dataclass(frozen=True)
class FastenerRef:
    body: BodyId
    index: int  # positional index within body's fastener list

@dataclass(frozen=True)
class WireRef:
    label: str  # matches WireRoute.label
```

**Migration plan:** `BodyId` and `JointId` should replace bare `str` in `Body.name` and `Joint.name` over time. This gives us:
1. Type-level distinction (a `BodyId` can't accidentally be used where a `JointId` is expected)
2. A single place to add validation (format rules, uniqueness checks) later
3. Consistent hashable keys for dicts and sets

This is the first use of the pattern; existing skeleton fields migrate incrementally.

#### Types

```python
class AssemblyAction(Enum):
    INSERT = "insert"             # place component into pocket/bracket
    FASTEN = "fasten"             # drive a fastener
    ROUTE_WIRE = "route_wire"     # run a cable through channel
    CONNECT = "connect"           # plug a connector into a port
    ARTICULATE = "articulate"     # move a joint to a specific angle


class ToolKind(Enum):
    HEX_KEY_2 = "hex_key_2mm"
    HEX_KEY_2_5 = "hex_key_2.5mm"
    HEX_KEY_3 = "hex_key_3mm"
    PHILLIPS_0 = "phillips_#0"
    PHILLIPS_1 = "phillips_#1"
    FINGERS = "fingers"
    TWEEZERS = "tweezers"
    PLIERS = "pliers"


@dataclass(frozen=True)
class ToolSpec:
    kind: ToolKind
    shaft_diameter: Meters
    shaft_length: Meters
    head_diameter: Meters          # clearance envelope
    grip_clearance: Meters         # lateral space needed for hand/fingers
    solid: Callable[[], Shape]     # geometry for visualization


@dataclass(frozen=True)
class AssemblyOp:
    step: int
    action: AssemblyAction
    target: ComponentRef | FastenerRef | WireRef | JointId  # typed union
    body: BodyId
    tool: ToolKind | None
    approach_axis: Vec3 | None     # derived from MountPoint.axis (negated) for FASTEN ops
    angle: Radians | None          # target angle, only for ARTICULATE ops
    prerequisites: tuple[int, ...]  # frozen — use tuple not list
    description: str               # human language — the only string


@dataclass
class AssemblyState:
    """Snapshot of what's physically present at a point in the assembly."""
    installed_components: frozenset[ComponentRef]
    installed_fasteners: frozenset[FastenerRef]
    routed_wires: frozenset[WireRef]
    joint_angles: dict[JointId, Radians]  # default: 0.0 for all joints; ARTICULATE ops set absolute angles


@dataclass
class AssemblySequence:
    ops: list[AssemblyOp]

    def state_at(self, step: int) -> AssemblyState:
        """Compute what's installed after completing step N.

        Replays ops 0..step, accumulating installed refs and joint angles.
        Joint angles start at 0.0; ARTICULATE ops set absolute values.
        """
        ...

    def geometry_at(self, step: int, body_solids: dict[BodyId, Shape]) -> Shape:
        """Union of geometry present at step N.

        Includes: printed body shells (always present from step 0),
        plus component/fastener solids for everything in state_at(step).
        Body shells include all cut-outs regardless of step — the cuts
        exist in the printed part. Only additive geometry (components,
        fasteners, wires) appears incrementally.
        """
        ...
```

Key properties:
- `state_at(step)` gives the DFM system the exact geometry context for each check.
- `approach_axis` on FASTEN ops is derived from `MountPoint.axis` (negated — tool approaches opposite to insertion direction), placed into body frame via the same transform used in Place.
- `angle` on ARTICULATE ops sets an absolute joint position as `Radians`. All joints start at 0.0.
- `geometry_at()` takes `dict[BodyId, Shape]` — the printed body shells. These are always present (they're what you print). Components, fasteners, and wires appear incrementally as their install steps complete. Body shells include all cut-outs from step 0 since cuts are baked into the print.
- The sequence is a partial order (DAG via prerequisites), not strictly linear.
- All ref types are frozen and hashable. No bare strings except `description`.

#### Tool Library

A small set of tool shapes for visualization and clearance checking:

| ToolKind | Shaft Diameter | Head Diameter | Shaft Length | Grip Clearance |
|----------|---------------|---------------|-------------|----------------|
| HEX_KEY_2 | 2mm | 2mm | 50mm | 20mm |
| HEX_KEY_2_5 | 2.5mm | 2.5mm | 50mm | 20mm |
| HEX_KEY_3 | 3mm | 3mm | 60mm | 25mm |
| PHILLIPS_0 | 4mm | 6mm | 80mm | 25mm |
| PHILLIPS_1 | 5mm | 8mm | 80mm | 25mm |
| FINGERS | 15mm | 15mm | 60mm | 30mm |
| TWEEZERS | 3mm | 8mm | 100mm | 20mm |
| PLIERS | 10mm | 25mm | 120mm | 30mm |

Each `ToolSpec.solid` generates actual geometry (hex key = cylinder + L-bend, tweezers = two tapered arms, etc.) for viewer rendering and boolean clearance checks.

#### Deriving the Sequence

The existing assembly guide (`botcad/emit/readme.py`) walks the kinematic tree and emits prose steps: mount components per body, attach children per joint. The structured `AssemblyOp` model is significantly richer — DAG ordering, approach axes, tool selection, wire routing, connector plugging. Building the sequence generator is a substantial design task, not a simple extraction from `readme.py`.

The approach: write a new `build_assembly_sequence(bot) -> AssemblySequence` function that walks the skeleton and component/fastener/wire data to emit typed ops. The ordering heuristics from `readme.py` (servos before brackets, components before wires) inform the prerequisites, but tool selection and approach axes come from fastener specs and `MountPoint.axis`.

For the connector-to-wire lookup: `WireRoute` carries `bus_type` and label but not connector type directly. The check must traverse `WirePort.connector_type` on the source component to find the `ConnectorSpec`. This lookup chain should be encapsulated in a helper on `WireRef` or in the check itself.

### 2. DFM Finding Data Model

```python
class DFMSeverity(Enum):
    ERROR = "error"       # physically impossible to build/print as-is
    WARNING = "warning"   # buildable but problematic
    INFO = "info"         # advisory


class DFMCheck(ABC):
    """Base class for all DFM checks. The check registry is built from subclasses."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine-readable name, used in finding IDs and API responses."""
        ...

    @abstractmethod
    def run(self, bot: Bot, sequence: AssemblySequence, body_solids: dict[BodyId, Shape]) -> list[DFMFinding]:
        ...


# Checks register by subclassing. The runner discovers all DFMCheck subclasses
# at import time — no central enum to maintain.
#
# Example:
#   class FastenerToolClearance(DFMCheck):
#       name = "fastener_tool_clearance"
#       def run(self, ...) -> list[DFMFinding]: ...


@dataclass(frozen=True)
class DFMFinding:
    check_name: str                            # matches DFMCheck.name of the check that produced it
    severity: DFMSeverity
    body: BodyId
    target: ComponentRef | FastenerRef | WireRef | JointId  # what triggered it
    assembly_step: int                         # which step this finding applies to
    title: str                                 # human language
    description: str                           # human language
    pos: Vec3                                  # world-space location
    direction: Vec3 | None                     # clearance cone axis, bend direction
    measured: Meters | Radians | None          # actual value
    threshold: Meters | Radians | None         # required value
    has_overlay: bool                          # whether an overlay mesh is available

    @property
    def id(self) -> str:
        """Deterministic, stable across runs: '{check_name}:{body}:{target}'."""
        return f"{self.check_name}:{self.body.name}:{self.target}"
```

Overlay meshes are served at `GET .../overlays/{finding.id}.stl` — the finding itself only carries `has_overlay: bool`, not a filename.

### 3. Tier 1 Checks

Each check takes body solids + component specs + `AssemblyState` at the relevant step. All data already exists in the pipeline.

#### 3a. Fastener Tool Clearance

For each fastener, at the assembly step where it's driven:
- Look up the required tool from fastener spec (M3 socket head = HEX_KEY_2_5)
- Place the tool envelope (cylinder: head_diameter radius, shaft_length height) at the fastener head, oriented along approach_axis
- Use ray-casting (sample points on the tool envelope, measure distance to nearest geometry) rather than OCCT boolean intersection — booleans on complex meshes risk hangs (see `SetFuzzyValue`/`SetUseOBB` history). Ray-casting is faster and more robust for clearance checking.
- Check against `geometry_at(step)` — only geometry installed so far
- Any ray hitting geometry within the tool envelope = error. Reports minimum clearance distance and obstructing body.
- Overlay mesh: the clearance cone, colored red where blocked.

#### 3b. Wire Channel vs Connector Head

For each wire route segment through a tube/channel:
- Compare channel cross-section to connector body dimensions (from `ConnectorSpec.body_dimensions`), not just wire diameter
- The connector must fit through at entry/exit points
- Violation = error. Reports channel size vs connector size.

#### 3c. Wire Bend Radius

For each wire route:
- Curvature is measured at segment junctions — the angle between consecutive `WireSegment` directions — not along a smooth spline (wire routes are piecewise-linear)
- Compare effective bend radius at each junction to minimum: 5x cable OD for static segments, 10x for joint-crossing segments
- Typical servo wire (AWG 26, ~1.5mm OD): static min 0.0075m, dynamic min 0.015m
- Violation = warning (tight) or error (physically impossible)
- Overlay mesh: highlight the tight segment, draw actual vs required radius arc.

#### 3d. Component Retention Audit

For each mounted component:
- Check: does it have fasteners attaching it, or a retention feature?
- A component in a pocket with only gravity = warning
- Component with mounting_points and driven fasteners = OK.

#### 3e. Connector Mating Access

For each connector port on a mounted component:
- Cast a ray along the mating axis (connector's `wire_exit_direction`)
- Check for 15mm clear along that axis, 10mm lateral clearance
- Check at the assembly step where the connection happens
- Obstruction = error. Reports what's blocking and available clearance.

### 4. API Surface

DFM analysis is expensive (geometry queries, boolean ops, ray casts). The API uses an async submit/poll/results pattern.

```
POST /api/bots/{bot}/dfm/run
  -> { run_id: str }

GET  /api/bots/{bot}/dfm/{run_id}/status
  -> {
      state: "running" | "complete" | "failed",
      checks_total: int,
      checks_complete: int,
      checks: [{ name: str, state: "pending" | "running" | "complete" | "failed", findings_count: int, error: str | null }]
     }

GET  /api/bots/{bot}/dfm/{run_id}/findings
  -> { findings: DFMFinding[] }    # everything discovered so far, grows as checks complete

GET  /api/bots/{bot}/dfm/{run_id}/overlays/{finding_id}.stl
  -> binary mesh (tool clearance cone, highlighted wire segment, etc.)

GET  /api/bots/{bot}/assembly-sequence
  -> { ops: AssemblyOp[], tool_library: ToolSpec[] }
```

The assembly sequence endpoint is separate from DFM — it's independently useful (powers assembly mode, replaces prose guide).

Checks run independently and report findings as they complete. The viewer can show results incrementally (fastener clearance finishes in 2s, you see those issues while bend radius is still computing).

### 5. Viewer Integration

New mode: **DFM Mode** (alongside Explore/Joints). Replaces Assembly mode, which is removed — the assembly step scrubber in DFM mode subsumes its functionality with better data.

**Prerequisite cleanup:** Remove `assembly-mode.ts` and its references before building DFM mode. This is an explicit first step in implementation.

**Panel:** Left sidebar shows a sortable, filterable **table** of findings. Columns: severity icon, check name, body, title, measured vs threshold. Sortable by any column. Filterable by severity and check name. Not a tree — findings are flat records, a table is more scannable.

**Click finding:** Camera flies to `finding.pos`, isolates the affected body, loads the overlay mesh (clearance cone in red, tight bend in orange, etc.).

**Assembly step scrubber:** Slider at the bottom. As you scrub, the bot builds up incrementally — only geometry from `state_at(step)` is visible. Findings filter to show only those relevant to steps 1 through current. You "watch" the assembly happen and see where it breaks down.

**Tool visualization:** When a finding involves tool access, the tool geometry renders at the approach position. You see exactly where the hex key would need to go and what's in the way.

**Progress indicator:** While DFM analysis is running, shows which checks are complete and streams findings as they arrive.

### 6. Sim Validation (Parallel Workstream)

Fully independent of the DFM system — no shared code, no dependencies. Included here for deadline coordination only. The wheeler_base has MuJoCo XML and meshes but driving behavior is unverified.

Validation criteria:
1. **Loads without error** — XML parses, meshes resolve, no collision warnings at init
2. **Drives straight** — equal wheel velocity produces forward motion along a line
3. **Turns** — differential speeds produce expected turning (left faster = turns right)
4. **Doesn't tip** — COM low enough that normal driving is stable
5. **Mass matches BOM** — sim total mass within 2% of 468g
6. **Ground contact stable** — wheels maintain contact, no jitter/bounce

Implementation: a test script (not a training env) that loads the scene, runs scripted maneuvers (forward 1s, turn left 1s, turn right 1s, stop), asserts position/orientation within tolerance, and generates a filmstrip for visual confirmation.

## Implementation Phases

### Phase 0: Foundational Cleanups

These are broad codebase improvements that the DFM work depends on. Complete before any DFM-specific code.

**0a. StringId pattern (`botcad/ids.py`):**
Introduce `BodyId` and `JointId`. Migrate `Body.name`, `Joint.name`, and all code that passes body/joint names as bare strings. This touches the skeleton, packing, emit, viewer manifest, and tests — but each change is mechanical (wrap/unwrap). The goal is that by the time DFM code is written, there are no bare-string entity names to get wrong.

**0b. Dimension types (`botcad/units.py`):**
Introduce `Meters` and `Radians` as `NewType` wrappers. Migrate existing dimension fields in component specs, fastener specs, connector specs, servo specs, and skeleton geometry. Again mechanical — wrap existing float values, update type annotations. Catches the class of bug where meters and millimeters get mixed (which has happened before in this codebase with the tool library table values).

**0c. Remove assembly mode:**
Delete `viewer/assembly-mode.ts` and its references. Clean break before building the DFM viewer mode that replaces it.

### Phase 1: Assembly Sequence Model
### Phase 2: DFM Check Framework + Tier 1 Checks
### Phase 3: API Surface
### Phase 4: Viewer DFM Mode
### Phase 5: Sim Validation (parallel with Phases 1-4)

## File Organization

```
botcad/
  ids.py                 # BodyId, JointId — the StringId pattern, shared across codebase
  units.py               # Meters, Radians NewType definitions
  assembly/
    sequence.py          # AssemblyAction, AssemblyOp, AssemblySequence, AssemblyState
    tools.py             # ToolKind, ToolSpec, tool library + geometry
    refs.py              # ComponentRef, FastenerRef, WireRef (compound refs using BodyId/JointId)
  dfm/
    check.py             # DFMCheck ABC, DFMSeverity, DFMFinding
    runner.py            # async check orchestration, run management, subclass discovery
    checks/              # each file defines a DFMCheck subclass — auto-discovered by runner
      fastener_clearance.py
      wire_channel_sizing.py
      wire_bend_radius.py
      component_retention.py
      connector_access.py

viewer/
  dfm-mode.ts            # DFM viewer mode (orchestration, overlay groups, activate/deactivate)
  dfm-panel.ts           # findings list panel (severity grouping, filtering, click-to-focus)
  assembly-scrubber.ts   # step slider + incremental geometry visibility

tests/
  test_dfm_wheeler_base.py   # regression: known findings for wheeler_base
  test_sim_validation.py      # Track C: driving behavior smoke tests
```

## What This Replaces

- `botcad/emit/readme.py` (prose assembly guide generation) — replaced by `AssemblySequence` rendered to language on demand
- `viewer/assembly-mode.ts` — removed, superseded by DFM mode's assembly step scrubber

## Out of Scope

- Tier 2 checks (overhang detection, wall thickness, enclosed cavities) — future work
- FEA integration into DFM findings — keep separate for now
- Training environment for wheeler_base — sim validation only, no RL
- Firmware / Pi software — deferred until physical bot exists
