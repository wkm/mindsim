# Learnings — 2026-03-21 — ShapeScript & Assembly Pipeline

## ShapeScript as an IR

- **A CAD IR between design code and the kernel is extremely valuable.** ShapeScript sits between Python parametric design and build123d/OCCT. Every piece of geometry becomes inspectable, cacheable, and debuggable. The initial investment was large but it pays dividends in every downstream feature.

- **The IR should be a language, not a recording.** CadStep (recording what OCCT did) vs ShapeScript (declaring what should be done) — ShapeScript won because it's the higher-level concept. You can reason about it, optimize it, and show it to users.

- **`Copy()` and `RadialArray()` as first-class ops dramatically improve readability.** Without them, 60 treads = 180 ops of noise. With them, it's one line. Adding domain-relevant ops to the IR is high leverage.

- **`PrebuiltOp` is an honest escape hatch.** When the IR can't express something (fillets needing edge selection, chamfers, hex socket extrusions), wrapping the build123d result as PrebuiltOp is better than faking it. Over time, add ops to eliminate PrebuiltOps — but don't block on having zero.

- **Snapshot tests on the IR text are cheap and powerful.** Golden-file `.shapescript` baselines catch any emitter change as a readable diff. Much faster than running OCCT to compare volumes.

## Assembly Hierarchy

- **Assembly > Body > Feature is the right taxonomy.** No "component" level between Assembly and Body. The bracket is a Feature of the parent body (fused in via CallOp), not a separate component. Servos, horns, batteries are purchased Bodies with their own ShapeScript.

- **Every physical object must be a Body.** Servos, horns, fasteners, mounted components — if it exists in the physical robot, it should be a Body in `bot.all_bodies` with `world_pos`, `world_quat`, and `shapescript`. This is the single source of truth that all exporters read from.

- **Assemblies should nest.** A gripper is an assembly inside an arm assembly inside a robot assembly. `Module` was too flat.

## Single Source of Truth

- **Multiple exporters computing transforms independently is a bug factory.** The horn rotation bug happened because STL export, STEP export, and MuJoCo XML each independently decided how to orient the horn. The fix: compute position/orientation ONCE during `solve()`, store on Body, all exporters read it.

- **The ShapeScript reference IS the geometry.** No Solid stored on Body — it's derived on demand from the ShapeScript. The content hash is the cache key. If the script hasn't changed, the cached BREP is returned instantly.

- **Caching at the ShapeScript level gives incremental rebuilds for free.** Change a servo on the arm → only that body's ShapeScript hash changes → only that body re-executes OCCT. Everything else is a cache hit.

## Viewer / Debugging

- **The ShapeScript debugger is the most useful development tool.** Split-pane: code editor (syntax-highlighted ops) on the right, 3D viewport on the left. Click a line to scrub. Outcome/Inputs view modes. Follow mode. This is how you debug geometry issues — you step through the program and see what each op does.

- **Real geometry for section caps, not stencil tricks.** Stencil-buffer section caps don't work reliably with EffectComposer post-processing. Computing actual triangulated cap geometry (contour segments → closed polygons → earcut triangulation with hole support) works everywhere. Key insight: you need a containment tree (outer/hole/island classification) for non-trivial cross-sections.

- **One Viewport3D component used everywhere.** All 3D views (bot viewer, component browser, ShapeScript debugger) use the same `Viewport3D` class. Orientation cube, measure tool, section plane, view presets, follow mode — added once, available everywhere.

- **The parts tree should show EVERYTHING.** Every servo, horn, fastener, wire, mounted component — in a searchable, filterable tree with category toggles. If it exists in the robot, it should be in the tree and clickable.

## Build123d / OCCT Gotchas

- **`.locate()` mutates in place; `.moved()` creates a copy.** On `@lru_cache`'d shapes, `.locate()` silently corrupts the cached object. Always use `.moved()`.

- **`Location(pos, euler)` applies rotation then translation independently — it does NOT orbit the position.** For circular patterns (wheel treads), you need two steps: `.moved(Location(pos))` then `.moved(Location((0,0,0), (0,0,angle)))`. A single `Location(pos, euler)` rotates the shape's orientation but doesn't move its position around the axis.

- **`THREE.Earcut` is not exposed on the namespace in newer Three.js.** Import it directly: `import { Earcut } from 'three/src/extras/Earcut.js'`.

- **Sequential boolean fuses cascade precision loss.** For 60 treads, do `clones[0].fuse(*clones[1:])` (one batch) instead of 59 sequential `result.fuse(clone)` calls.

## Process

- **Agent teams with parallel tasks work well for independent refactors.** PrebuiltOp elimination (4 parallel agents), emitter refactors (3 parallel agents) — each agent has its own scope, no conflicts.

- **The /simplify review pattern catches real issues.** Three parallel review agents (reuse, quality, efficiency) found: variable shadowing bug in RadialArrayOp, O(N²) tag lookup, 150 lines of dead code, missing batch fuse optimization.

- **Regenerate all models after pipeline changes.** Stale meshes cause confusion. The web server now auto-regenerates when `design.py` is newer than the manifest — but only on `make web`, not on `make`.
