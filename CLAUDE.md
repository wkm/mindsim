# MindSim - Parametric Robot Design → Sim → Fabrication

Design robots from real components, train them in simulation, then build the physical thing.

## Pipeline Philosophy

The parametric skeleton is the **single source of truth**. One design produces everything — simulation, printable parts, assembly instructions, and training environments.

```
Skeleton DSL (design.py)
    ├─→ CAD assembly (STEP + per-body STLs for slicing)
    ├─→ MuJoCo XML (references same STL meshes — no separate "sim geometry")
    ├─→ validation renders (visual sanity checks before printing)
    └─→ training pipeline (PPO, reward shaping, scene gen)
```

**Key invariants:**

- **CAD geometry = sim geometry.** MuJoCo references the same STLs you'd send to a slicer.
- **Rigid sub-chains become single printed parts.** Servos are the mechanical separation points.
- **Purchased components get pockets, not models.** Assembly guide says what to insert where.
- **Sim fidelity matters.** Geometry, mass, and actuation should match physical reality.

## Quick Start

```bash
uv run mjpython main.py                    # Interactive TUI (default)
uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython main.py train [--smoketest] [--bot NAME]
uv run mjpython main.py play [CHECKPOINT] [--bot NAME] [--run RUN_NAME]
uv run mjpython main.py scene              # Scene gen preview
```

`--bot NAME`: bot directory name (e.g., `wheeler_arm`). Make shortcuts: `make`, `make view`, `make train`.

**Worktrees:** `make wt-new NAME=foo`, `make wt-new NAME=bar TYPE=infra`, `make wt-ls`, `make wt-rm NAME=foo`

## Project Structure

```
├── main.py                   # Single entry point for all modes
├── botcad/                   # Parametric bot CAD system
│   ├── skeleton.py           # Kinematic tree DSL (Bot, Body, Joint)
│   ├── component.py          # Component base classes (ServoSpec, etc.)
│   ├── components/           # Real component catalog (STS3215, RPi, wheels, etc.)
│   ├── bracket.py            # Parametric servo bracket geometry
│   ├── geometry.py           # Servo placement, quaternion math
│   ├── packing.py            # Body dimension solver
│   ├── routing.py            # Wire route solver
│   └── emit/                 # Output generators (cad, mujoco, bom, readme, renders)
├── bots/                     # Bot definitions + generated outputs
│   └── wheeler_arm/
│       ├── design.py         # Bot definition (the source of truth)
│       └── meshes/*.stl      # Generated meshes (used by sim + slicer)
├── sim_env.py                # MuJoCo simulation wrapper (SimEnv)
├── train.py                  # PPO training loop & policy networks
├── scene_gen/                # Procedural scene generation (auto-discovered concepts)
├── worlds/room.xml           # Static arena (floor, curbs, target)
└── runs/                     # Per-run directories (gitignored)
```

Each bot's `design.py` calls `bot.solve()` then `bot.emit()` to produce: `assembly.step`, `meshes/*.stl`, `bot.xml`, `scene.xml`, `bom.md`, `assembly_guide.md`, and validation renders.

## Branching & Change Philosophy

1. **Master stays stable.** Only small, safe changes land directly on master.
2. **Experiments are focused.** `exp/YYMMDD-<name>` branches test one hypothesis. No drive-by refactors.
3. **Tooling/infra changes are separate.** `infra/<name>` branches, not mixed into experiments.
4. **Commit logs are a journal.** Explain *why*, not just *what*.

**Before any non-trivial change:** ensure working tree is clean, classify the change, use `make wt-new` for experiments/infra.

**Experiment workflow:** `make wt-new NAME=foo` → work on branch → add entry to `EXPERIMENTS.md` → merge or abandon.

**Tracking:** `EXPERIMENTS.md` records branch, hypothesis, status, outcome, W&B links.

## Future Directions

- **Structural validation (FEA):** Run finite-element analysis on generated meshes to validate printability before fabrication. The immediate smoke test is cross-section analysis (servo stall torque → minimum bridge cross-section at each bracket). The longer-term goal is mesh FEA using `solidspy` or `sfepy` — load each body with servo reaction forces and gravity, check peak stress vs. material yield strength. This would catch thin bridges, under-supported brackets, and other structural failures that the packing solver doesn't reason about.

## Development Notes

- **Clean up before committing** — remove debug*.py, test*.py, .rrd files, temp files
- **Bot changes require the checklist** — read `NEW_BOT_CHECKLIST.md` before committing bot XML changes

## Commit Message Format

```
Short summary (imperative mood, <72 chars)

Why this change was made. Context, reasoning, alternatives considered,
observations. The git log is the project's journal.

Session: <link to Claude session>
```
