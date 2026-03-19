# MindSim

Design robots from real components, train them in simulation, build the physical thing.

## Principles

- Simple is beautiful
- Composable modules
- The parametric skeleton is the **single source of truth**. One design produces everything — simulation, printable parts, assembly instructions, and training environments.
- **Derive from geometry, don't approximate.** When build123d/OCCT can compute a property (mass, COM, inertia, surface area) from actual CAD solid geometry, use it. The CAD solid is ground truth. Heuristic estimates are fallbacks, not the primary path.
- **Sim fidelity matters.** Geometry, mass, and actuation should match physical reality. CAD geometry = sim geometry — MuJoCo references the same STLs you'd send to a slicer.
- **One body, one mesh.** Each body solid is the union of its structural shell and all attachment geometry (brackets, cradles, couplers). That single mesh is both the collision and visual representation in simulation — no layered overlays. If geometry belongs to a body, union it into the body solid; don't bolt on a second visual-only mesh.
- **Design / Compose / Place / Cut.** The component geometry pipeline has four stages:
  1. **Design** — build each primitive (servo, bracket, coupler, clearance envelope) in its own local frame. The clearance/cut solid is a design-time artifact, visible in component renders.
  2. **Compose** — assemble related primitives (bracket wraps servo, coupler mates shaft) in local frame.
  3. **Place** — one rigid transform per component instance into the parent body frame.
  4. **Cut** — apply the pre-designed clearance solid to the parent body, using the **same transform** as Place.

  The critical invariant: Place and Cut use identical transforms. No axis-sign-dependent logic, no extra shifts at cut time. If a clearance shape needs asymmetry (e.g. outboard-only tolerance), that asymmetry is designed into the cut solid in step 1, in local frame, where it can be seen and validated.
- **`.moved()`, never `.locate()`.** build123d's `.locate()` mutates in place and returns `self`. On `@lru_cache`d solids (bracket, envelope, coupler, servo), this silently corrupts the cached object for all future callers. Always use `.moved()` which creates an independent copy. The CAD steps debug viewer (`?cadsteps=bot:body`) makes this kind of bug visible.
- **Commit logs are a journal.** Explain _why_, not just _what_.

## Quick Start

```bash
uv run mjpython main.py                    # Interactive TUI
uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython main.py web [--bot NAME]   # Web viewer (3D, components, CAD steps)
uv run mjpython main.py train [--bot NAME]
uv run mjpython main.py scene              # Scene gen preview
```

`--bot NAME`: bot directory name (e.g., `wheeler_arm`). Shortcuts: `make`, `make view`, `make train`.

## Development

- **Run `make validate` after every major step** — lint + tests + render regeneration. Review render diffs before committing.
- **Worktrees for experiments:** `make wt-new NAME=foo` → `exp/YYMMDD-foo` branch. Track in `EXPERIMENTS.md`.
- **Bot changes require `NEW_BOT_CHECKLIST.md`.**
- **TUI changes require snapshot tests:** `uv run pytest tests/test_tui_snapshots.py -v`
