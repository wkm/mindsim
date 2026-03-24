# Wire Stub & Fastener Visualization

**Date:** 2026-03-24
**Status:** Approved
**Workstream:** 3 of 3

## Problem

Components have detailed connector (`WirePort`) and mounting (`MountPoint`) metadata. Wire routing is fully computed (`solve_routing()`). Fastener geometry exists (`fastener_solid()`). But none of this renders in the viewer. Users can't verify that brackets don't block cable exits, fastener heads clear adjacent geometry, or wiring fits within component envelopes.

## Scope

This spec covers:
1. Wire stubs at connector sockets (visual collision check)
2. Fastener instances at mount points (bracket/servo/coupler/horn)

Out of scope (future): full cable routing visualization, assembly tool clearance validation.

## Relationship to Existing Manifest Data

The current `emit_viewer_manifest()` already emits fasteners and wire segments into the `parts` list with `category: "fastener"` and `category: "wire"`. This spec **extends those existing entries** with the additional fields needed for 3D rendering (position, direction, axis transforms). No new top-level arrays — wire stubs and fastener instances are enriched versions of existing `parts` entries.

The existing wire `parts` come from `solve_routing()` and represent full routed segments. Wire stubs are a separate concept — short segments at connector sockets, not full cable routes. Both are emitted as `parts` with `category: "wire"` but distinguished by a `wire_kind` field (`"stub"` vs `"route"`).

## Design

### Wire Stubs

**What:** Short wire segments (~20-30mm) extending from each `WirePort` along the connector's wire exit direction. Enough to see "there's a wire here, does it collide?"

**Data flow — direction lookup:**
`WirePort.connector_type` → `connector_spec()` lookup → `ConnectorSpec.wire_exit_direction` → transform by mount placement (mount.resolved_pos + mount rotation) + component-local port position → body-local frame.

**Manifest entry** (extends existing `parts` with `category: "wire"`):

```json
{
  "id": "wire_stub_uart_in_servo_shoulder",
  "name": "UART In",
  "category": "wire",
  "wire_kind": "stub",
  "body": "arm",
  "position": [x, y, z],
  "direction": [dx, dy, dz],
  "length": 0.025,
  "bus_type": "uart_half_duplex",
  "connector_type": "5264_3pin"
}
```

Position and direction are in body-local frame.

**Handling empty `connector_type`:** If a `WirePort` has no `connector_type` (empty string), skip the stub — no connector means no physical wire exit to visualize.

**Viewer side:**
- Render as a small cylinder (2-3mm diameter, length from entry)
- Color by bus type:
  - `UART_HALF_DUPLEX` = blue
  - `CSI` = green
  - `POWER` = red
  - `USB` = white
  - `GPIO` = purple
  - `PWM` = orange
  - `BALANCE` = yellow
  - Fallback = gray

### Fastener Instances

**What:** Render screw instances at every `MountPoint` in their inserted position.

**Transform chains** (component-local → body-local):

**(a) Component mount points** (e.g., Pi case screws, camera mounts):
`MountPoint.pos` + `MountPoint.axis` → rotate by mount rotation (face_outward, rotate_z) → translate by `mount.resolved_pos` → body-local frame.

**(b) Servo mounting ear screws** (bracket ↔ servo):
`MountPoint.pos` (ear position in servo-local frame) → rotate by servo joint placement → translate by joint position → body-local frame. The bracket emitter already computes these positions; emit them during manifest generation.

**(c) Horn mounting screws** (coupler ↔ horn):
`MountPoint.pos` (horn hole in servo-local frame) → rotate by servo joint placement → body-local frame. These attach the coupler to the horn.

**(d) Rear horn mounting screws:**
Same as (c) but on the blind-side horn face.

**Manifest entry** (extends existing `parts` with `category: "fastener"`):

```json
{
  "id": "fastener_servo_ear_1",
  "name": "M3x8 SHC",
  "category": "fastener",
  "body": "arm",
  "position": [x, y, z],
  "axis": [ax, ay, az],
  "mesh": "hardware_M3_shc.stl",
  "material": "steel",
  "context": "bracket ear -> servo case"
}
```

Position and axis in body-local frame. Context string for tooltip.

**Fastener STL generation:** The existing build pipeline generates fastener STLs via `fastener_stl_stem()` → `fastener_solid()`. These are already referenced in the current manifest. No new generation step needed.

**Viewer side:**
- Load fastener STL meshes (shared across instances via Three.js instancing)
- Position at each mount point with rotation aligning mesh Z-axis to fastener axis
- Material from workstream 1 catalog (`MAT_STEEL`, `MAT_NICKEL`)
- If workstream 1 isn't merged yet, use hardcoded metallic gray (`color: #888`, `metalness: 0.8`, `roughness: 0.3`)

### Viewer Integration

**Overlay layers in component tree (uses workstream 2 Layer nodes):**
- "Wires" layer — toggle all wire stubs
- "Fasteners" layer — toggle all fastener instances
- If workstream 2 isn't merged yet, use simple checkbox toggles in toolbar

**Interaction:**
- Hover wire stub → tooltip: "{name} — {bus_type}"
- Hover fastener → tooltip: "{name} — {context}"
- When a component is isolated, show only its wire stubs and fasteners
- Wire stubs and fasteners participate in the visibility model (hidden when parent body is hidden)

### Connector Housing (stretch goal)

If time allows, render the actual connector housing shape at the wire port:
- `connector_solid()` and `receptacle_solid()` already generate geometry
- Export as STLs alongside fastener meshes
- Place at wire port position, oriented by mating direction

## Validation

1. **Manifest schema test:** Snapshot test verifying new fields appear on wire/fastener `parts` entries for the test bot
2. **Transform correctness:** Fasteners should appear centered on mount holes, wire stubs should exit from connector faces. Visual check via `make validate` render diffs.
3. **Viewer toggle:** Verify wires/fasteners toggle on/off independently and respect body hide/isolate
4. **Bus type coverage:** Verify all `BusType` enum values have assigned colors (no fallback gray in normal use)

## Files Changed

| File | Change |
|------|--------|
| `botcad/emit/viewer.py` | Extend wire/fastener `parts` entries with position, direction, axis, transforms |
| `botcad/connectors.py` | (stretch) Export connector STLs |
| `viewer/bot-scene.ts` | Wire/fastener overlay visibility state |
| `viewer/bot-viewer.ts` | Load and instance wire stub + fastener meshes |
| `viewer/scene-sync.ts` | Wire/fastener visibility tied to parent body visibility |
| `viewer/component-tree.ts` | Wire/Fastener layer toggles |

## Dependencies

- **Light dependency on workstream 1** (materials): Fastener materials. Can stub with hardcoded metallic gray if not merged yet.
- **Light dependency on workstream 2** (tree): Layer nodes for toggles. Can stub with toolbar checkboxes if not merged yet.
