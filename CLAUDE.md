# MindSim

Design robots from real components, train them in simulation, build the physical thing.

## Principles

- Simple is beautiful
- Composable modules
- The parametric skeleton is the **single source of truth**. One design produces everything — simulation, printable parts, assembly instructions, and training environments.
- **Derive from geometry, don't approximate.** When build123d/OCCT can compute a property (mass, COM, inertia, surface area) from actual CAD solid geometry, use it. The CAD solid is ground truth. Heuristic estimates are fallbacks, not the primary path.
- **Sim fidelity matters.** Geometry, mass, and actuation should match physical reality. CAD geometry = sim geometry — MuJoCo references the same STLs you'd send to a slicer.
- **Commit logs are a journal.** Explain _why_, not just _what_.

## Quick Start

```bash
uv run mjpython main.py                    # Interactive TUI
uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython main.py train [--bot NAME]
uv run mjpython main.py scene              # Scene gen preview
```

`--bot NAME`: bot directory name (e.g., `wheeler_arm`). Shortcuts: `make`, `make view`, `make train`.

## Development

- **Run `make validate` after every major step** — lint + tests + render regeneration. Review render diffs before committing.
- **Worktrees for experiments:** `make wt-new NAME=foo` → `exp/YYMMDD-foo` branch. Track in `EXPERIMENTS.md`.
- **Bot changes require `NEW_BOT_CHECKLIST.md`.**
- **TUI changes require snapshot tests:** `uv run pytest tests/test_tui_snapshots.py -v`
