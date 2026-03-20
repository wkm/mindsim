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
- **Commit logs are a journal.** Explain _why_, not just _what_.

## Quick Start

```bash
uv run mjpython main.py                    # Interactive TUI
uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython main.py train [--bot NAME]
uv run mjpython main.py scene              # Scene gen preview
```

`--bot NAME`: bot directory name (e.g., `wheeler_arm`). Shortcuts: `make`, `make view`, `make train`.

## ShapeScript (Intermediate Representation)

The CAD pipeline has an optional ShapeScript layer between parametric design code and build123d/OCCT. It enables caching, swappable backends, and step-by-step visual debugging.

```bash
SHAPESCRIPT=1 uv run mjpython main.py view --bot wheeler_arm  # Use ShapeScript path
uv run python main.py shapescript-debug --bot wheeler_arm --body base  # Step-by-step Rerun debugger
```

- **Enable:** Set `SHAPESCRIPT=1` env var. Direct build123d path remains the default.
- **Debug:** `shapescript-debug` subcommand builds a body via ShapeScript and launches Rerun with per-op mesh snapshots.
- **Architecture:** `emit_body_ir()` translates `_make_body_solid()` logic into typed ShapeScript ops (`botcad/shapescript/ops.py`). `OcctBackend` executes the ShapeScript against build123d. Bracket/component solids are injected as `PrebuiltOp`.
- **Key files:** `botcad/shapescript/emit_body.py`, `botcad/shapescript/backend_occt.py`, `botcad/shapescript/program.py`, `botcad/shapescript/ops.py`

## Development

- **Run `make validate` after every major step** — lint + tests + render regeneration. Review render diffs before committing.
- **Worktrees for experiments:** `make wt-new NAME=foo` → `exp/YYMMDD-foo` branch. Track in `EXPERIMENTS.md`.
- **Bot changes require `NEW_BOT_CHECKLIST.md`.**
- **TUI changes require snapshot tests:** `uv run pytest tests/test_tui_snapshots.py -v`
