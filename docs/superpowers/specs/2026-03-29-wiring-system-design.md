# Wiring System: Logical Model + Diagram View

## Context

The bot's wiring connectivity is currently implicit — hard-coded in `solve_routing()` functions that know which ports connect (`_route_servo_bus`, `_route_camera_csi`, etc.). There's no inspectable data structure representing "what connects to what" independent of physical routing. This makes it hard to validate completeness (are all ports wired?) and correctness (are the right things connected?) without reading routing code.

We need:
1. An explicit logical wiring model (`WireNet`) that captures connectivity as data
2. A wiring diagram view in the Assembly tab to visualize and validate that model
3. Eventually, `solve_routing()` consumes `WireNet[]` instead of hard-coding connectivity

## Phase 1: WireNet Data Model

### Data Structures

All new types live in `botcad/wirenet.py`.

A new `ComponentId` type is added to `botcad/ids.py`, following the existing `BodyId`/`JointId` pattern (str subclass).

```python
# botcad/ids.py
class ComponentId(str):
    """Unique identifier for a component instance in the skeleton."""
    ...

# botcad/wirenet.py
class NetTopology(StrEnum):
    DAISY_CHAIN = "daisy_chain"    # ordered chain: ports[0] → [1] → [2] → ...
    POINT_TO_POINT = "point_to_point"  # exactly 2 ports
    STAR = "star"                  # ports[0] is source, rest are sinks

@dataclass(frozen=True)
class NetPort:
    """A specific port on a specific component instance."""
    component_id: ComponentId  # which component instance
    port_label: str            # matches WirePort.label on that component

@dataclass(frozen=True)
class WireNet:
    """A named logical connection between ports.

    Ports are listed at hop level — each port in the tuple is an
    individual endpoint. For DAISY_CHAIN, adjacent pairs form hops:
    ports[0]→ports[1], ports[1]→ports[2], etc.
    """
    label: str              # "servo_bus", "camera_csi", "power_pi"
    bus_type: BusType
    topology: NetTopology
    ports: tuple[NetPort, ...]
```

### Key Design Decisions

- **Hop-level ports, not component-level.** A daisy-chain servo bus lists `(waveshare/uart_out, shoulder/uart_in, shoulder/uart_out, elbow/uart_in)` — not just `(waveshare, shoulder, elbow)`. This enables per-port validation (overloaded ports, missing connections) and matches the physical wiring spec.
- **ComponentId, not BodyId.** A component might move between bodies during design. The logical wiring doesn't care where a component is mounted, only that port A connects to port B.
- **Frozen dataclass.** Consistent with project conventions.

### Derivation

A new function `derive_wirenets(bot: Bot) -> tuple[WireNet, ...]` produces the net list. Initially, the connectivity knowledge that's currently hard-coded in `solve_routing()` (e.g., "servos form a UART daisy-chain," "camera CSI connects to Pi") moves into `derive_wirenets()` as explicit `WireNet` construction. The logic is the same — walk the skeleton, find servos, find camera, etc. — but the output is a declarative net list instead of physical segments.

Long-term, bots could declare their wiring spec declaratively (e.g., as data on the bot definition), but that's out of scope. For now, `derive_wirenets()` is the single place that encodes "what connects to what."

The net list is stored on `Bot.wire_nets: list[WireNet]` alongside the existing `Bot.wire_routes`.

### Validation Checks

New DFM checks in `botcad/dfm/checks/`:

- **`wirenet_orphaned_ports.py`** — Ports declared on components that appear in zero nets. Severity: warning.
- **`wirenet_overloaded_ports.py`** — A port that appears in multiple nets (unless bus type allows it). Severity: error.
- **`wirenet_bus_type_mismatch.py`** — A net's bus_type doesn't match the bus_type declared on the connected WirePorts. Severity: error.

### API Endpoint

`GET /api/bots/{name}/wirenets` returns the `WireNet[]` as JSON for the viewer.

## Phase 2: Wiring Diagram View

### Library Choice: elkjs + custom SVG

The diagram uses [elkjs](https://github.com/kieler/elkjs) (EPL-2.0) for port-aware automatic layout, with hand-rendered SVG for display and interaction. This matches the pattern used by the existing `assembly-dag.ts`.

**Why elkjs directly (not Sprotty):**
A spike confirmed that Sprotty's port-to-port edge routing works, but the full Sprotty stack pulls in inversify (DI container), snabbdom (virtual DOM), and reflect-metadata — heavy dependencies for what is essentially a static diagram. Using elkjs directly for layout + custom SVG keeps the code simple and consistent with existing viewer patterns.

### Graph Model Mapping

```
WireNet[] → GraphNode[] + GraphEdge[] → elkjs layout → SVG

Each unique component_id  → GraphNode (ElkNode)
  - label: component name
  - ports: input ports on left (WEST), output ports on right (EAST)

Each NetPort              → GraphPort (ElkPort) on its parent node
  - label: port_label
  - side: determined by edge direction (target=WEST, source=EAST)

Adjacent port pairs       → GraphEdge (ElkExtendedEdge, port-to-port)
  - DAISY_CHAIN: ports[i] → ports[i+1], skipping same-component pairs
  - POINT_TO_POINT: ports[0] → ports[1]
  - STAR: ports[0] → ports[i] for i in 1..n
  - Colored by bus_type
```

This transform lives in `viewer/wiring-diagram.ts` — a pure function from `WireNet[]` to the internal graph model, then to elkjs input, then to SVG.

### Viewer Integration

- **Location:** New Assembly sub-tab "Wiring" alongside Steps/DAG/DFM
- **Layout:** Sprotty container fills the nav pane (top-left), same pattern as DAG tab
- **Data source:** Fetches from `/api/bots/{name}/wirenets`
- **Detail pane (bottom-left):** Shows selected node/edge details
  - Click component → port list with connection status (connected / orphaned)
  - Click edge → net label, bus type, topology, connected ports

### Interaction

- Click node: highlight all connected edges, show port details
- Click edge: highlight full net, show net details
- Bus type legend in corner
- No 3D viewport interaction (purely 2D logical diagram)
- Read-only — no editing

### Validation Indicators

Rendered directly on the diagram:

- **Orphaned port:** orange hollow circle on the SPort
- **Port in multiple nets:** red outline on the SPort
- **Bus type mismatch:** red dashed edge

These correspond to the DFM checks from Phase 1 — the diagram makes them visible spatially.

### Styling

- Node borders colored by component kind (servo=purple, compute=green, etc.)
- Edge color by bus type, matching the existing `_BUS_TYPE_COLORS` in `botcad/emit/viewer.py`:
  - UART: `rgb(51, 153, 219)` (blue) — `[0.20, 0.60, 0.86]`
  - CSI: `rgb(102, 186, 107)` (green) — `[0.40, 0.73, 0.42]`
  - Power: `rgb(230, 77, 64)` (red) — `[0.90, 0.30, 0.25]`
  - Fallback for unmapped types (PWM, USB, GPIO, BALANCE): `rgb(135, 135, 135)` (gray) — `[0.53, 0.53, 0.53]`
- CSS via Sprotty's class-based styling system

## Phase 3: solve_routing() Consumes WireNets (deferred)

Deferred until Phases 1-2 validate the logical model. **Phases 1-2 must not modify `solve_routing()` or `WireRoute`** — the existing physical routing continues to work independently. The refactor:

- `solve_routing(bot)` reads `bot.wire_nets` instead of hard-coding per-bus functions
- Each `WireRoute` gains a `net: WireNet` back-reference
- The per-bus routing functions (`_route_servo_bus`, etc.) become topology-aware: a `DAISY_CHAIN` net routes along the kinematic tree visiting components in order; a `POINT_TO_POINT` net finds the shortest path between two components.

## Files Modified

### Phase 1 (Python)
- **Modified:** `botcad/ids.py` — add `ComponentId`
- **New:** `botcad/wirenet.py` — `WireNet`, `NetPort`, `NetTopology`, `derive_wirenets()`
- **New:** `botcad/dfm/checks/wirenet_orphaned_ports.py`
- **New:** `botcad/dfm/checks/wirenet_overloaded_ports.py`
- **New:** `botcad/dfm/checks/wirenet_bus_type_mismatch.py`
- **Modified:** `botcad/skeleton.py` — add `Bot.wire_nets` field
- **Modified:** `botcad/emit/api.py` — add `/api/bots/{name}/wirenets` endpoint
- **New:** `tests/test_wirenet.py` — unit tests for derivation + validation

### Phase 2 (TypeScript)
- **New:** `viewer/wiring-diagram.ts` — Sprotty container, WireNet→SGraph transform
- **Modified:** `viewer/assembly-viewer.ts` — add "Wiring" sub-tab
- **Modified:** `package.json` — add `elkjs` dependency

### Phase 3 (Python, deferred)
- **Modified:** `botcad/routing.py` — consume `WireNet[]`
- **Modified:** `botcad/routing.py` — add `net` field to `WireRoute`

## Verification

### Phase 1
- `make lint` passes
- `uv run pytest tests/test_wirenet.py` — derivation produces correct nets for wheeler_arm bot
- DFM checks fire correctly: introduce an intentional orphaned port, confirm warning
- `curl /api/bots/wheeler_arm/wirenets` returns valid JSON matching the data model

### Phase 2
- Sprotty spike: port-to-port edges render correctly with elkjs layout
- `pnpm exec tsc --noEmit` passes
- `pnpm exec biome check viewer/` passes
- Wiring tab renders wheeler_arm nets as a block diagram with correct colors and port labels
- Click interaction shows detail pane content
- Orphaned port indicator visible when a port is unconnected
