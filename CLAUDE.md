# MindSim - Parametric Robot Design → Sim → Fabrication

Design robots from real components, train them in simulation, then build the physical thing.

## Principles

- Simple is beautiful
- Composable modules

## Pipeline Philosophy

The parametric skeleton is the **single source of truth**. One design produces everything — simulation, printable parts, assembly instructions, and training environments.

```md
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

```txt
mindsim/
├── main.py                      # Entry point + TUI app (MindSimApp)
│
├── botcad/                      # Parametric bot CAD system
│   ├── skeleton.py              # Kinematic tree DSL (Bot, Body, Joint)
│   ├── component.py             # Component base classes (ServoSpec, etc.)
│   ├── components/              # Real component catalog (STS3215, RPi, wheels, etc.)
│   ├── bracket.py               # Parametric servo bracket geometry
│   ├── geometry.py              # Servo placement, quaternion math
│   ├── packing.py               # Body dimension solver
│   ├── routing.py               # Wire route solver
│   ├── validation.py            # Subassembly ROM sweep + collision detection
│   └── emit/                    # Output generators (cad, mujoco, bom, readme, renders)
│       ├── render3d.py          # Centralized 3D rendering (SceneBuilder, Renderer3D, Color)
│       ├── composite.py         # Image compositing (grid, filmstrip, fonts)
│       ├── component_renders.py # Component/bracket tear sheets
│       ├── renders.py           # Bot-level overview, closeups, sweep filmstrips
│       └── assembly_renders.py  # Assembly instruction PDFs
│
├── training/                    # RL training loop
│   ├── train.py                 # Main training orchestration
│   ├── env.py                   # TrainingEnv (wraps SimEnv with rewards)
│   ├── pipeline.py              # Per-bot training config
│   ├── algorithms.py            # PPO/REINFORCE
│   ├── collection.py            # Episode collection
│   ├── parallel.py              # Multiprocessing collection
│   ├── policies.py              # Neural networks (LSTM, Tiny)
│   ├── rewards.py               # Reward hierarchy
│   ├── checkpoint.py            # Save/load checkpoints
│   ├── dashboard.py             # Metrics display helpers
│   ├── tweaks.py                # Live config (tweaks.json)
│   ├── run_manager.py           # Run lifecycle, naming, W&B init
│   └── git_utils.py             # Git helpers
│
├── sim/                         # Physical simulation
│   ├── env.py                   # SimEnv (MuJoCo wrapper)
│   └── scene_preview.py         # Scene gen preview viewer
│
├── viz/                         # Visualization & replay
│   ├── replay.py                # Download/regenerate recordings
│   ├── rerun_logger.py          # Rerun logging
│   ├── rerun_wandb.py           # W&B integration
│   ├── visualize.py             # One-shot visualization
│   └── blueprint.py             # Rerun layout
│
├── tools/                       # Standalone tools
│   ├── play.py                  # Interactive play
│   ├── view.py                  # MuJoCo viewer
│   ├── quick_sim.py             # Debug viz
│   ├── stability_test.py        # Stability testing
│   ├── remote_train.py          # GCP runner
│   └── ai_commentary.py         # AI analysis
│
├── tui/                         # TUI screens
│   └── screens/
│       ├── main_menu.py         # Top-level menu
│       ├── run_browser.py       # Browse training runs
│       ├── run_action.py        # Actions for a selected run
│       ├── gcp_instances.py     # GCP instance management
│       ├── bot_selector.py      # Bot picker
│       ├── dirty_tree.py        # Uncommitted changes warning
│       ├── checkpoint_picker.py # Checkpoint selection
│       └── training_dashboard.py # Live training metrics
│
├── scene_gen/                   # Procedural scene generation
│   ├── primitives.py            # Prim types, GeomType, materials
│   ├── composer.py              # SceneComposer, placement
│   └── concepts/                # Parametric furniture (auto-discovered)
│
├── bots/                        # Bot definitions + generated outputs
│   ├── simple2wheeler/
│   │   ├── bot.xml              # Robot: bodies, joints, cameras, meshes
│   │   ├── scene.xml            # Thin wrapper: timestep + bot.xml + room.xml
│   │   └── meshes/*.stl         # Visual geometry
│   └── wheeler_arm/
│       ├── design.py            # Bot definition (the source of truth)
│       └── meshes/*.stl         # Generated meshes (used by sim + slicer)
│
├── worlds/
│   └── room.xml                 # Standalone arena (floor, curbs, target)
│
├── tests/                       # Test suite
│   ├── test_smoketest.py        # 39 training/env tests
│   ├── test_tui_snapshots.py    # 4 TUI snapshot tests
│   └── snapshot_drivers/        # Minimal apps for snapshot tests
│
└── runs/                        # ALL runtime output (gitignored)
    ├── <run_name>/              # Training runs
    │   ├── run_info.json
    │   ├── checkpoints/
    │   └── recordings/
    └── replays/                 # Downloaded recordings
        └── <run_name>/
```

Each bot's `design.py` calls `bot.solve()` then `bot.emit()` to produce: `assembly.step`, `meshes/*.stl`, `bot.xml`, `scene.xml`, `bom.md`, `assembly_guide.md`, and validation renders.

## Environment API

```python
from sim.env import SimEnv

env = SimEnv(render_width=128, render_height=128)

# Step simulation
camera_img = env.step(left_motor=0.5, right_motor=0.5)  # [-1, 1]

# Access MuJoCo model/data directly
env.model    # MjModel - structure (bodies, geoms, cams, meshes)
env.data     # MjData - state (xpos, xquat, sensor data)

# Convenience methods
env.get_bot_position()
env.get_target_position()
env.get_distance_to_target()

env.close()
```

## Visualization

```bash
uv run mjpython main.py visualize
rerun recordings/simple2wheeler_viz.rrd
```

Creates a Rerun recording with:

- 3D robot meshes from MuJoCo model
- Camera view with proper FOV
- Trajectory trail
- Motor inputs and distance plots

## Procedural Scene Generation

The `scene_gen/` module generates rooms with parametric furniture for training diversity.

**Preview scenes:**

```bash
uv run mjpython main.py scene              # MuJoCo viewer with random furniture
```

Loads `worlds/room.xml` directly — no bot needed. Controls: `Space` = next scene, `Backspace` = regenerate same scene, `Arrow keys` = move target.

**Architecture:**

1. **Concepts** (`scene_gen/concepts/`) — Each concept is a Python file with a frozen `Params` dataclass + `@lru_cache`d `generate()` function that returns MuJoCo primitives. Auto-discovered via `pkgutil`.
2. **Primitives** (`scene_gen/primitives.py`) — `Prim` dataclass mapping to MuJoCo geom types (box, cylinder, sphere). Includes material color constants.
3. **Composer** (`scene_gen/composer.py`) — `SceneComposer` discovers pre-allocated obstacle slot bodies in the MuJoCo model, writes concept primitives into them at reset time. No model recompilation needed.

**Scene identity:** Each scene has an integer `seed` for deterministic reproduction. `describe_scene()` produces a human-readable text description. `scene_id()` gives a short hex tag.

**Adding new furniture:** Drop a file in `scene_gen/concepts/` following the pattern in `concepts/__init__.py` docstring. Needs a `Params` frozen dataclass + `generate(params) -> tuple[Prim, ...]`. That's it — auto-discovered.

**Obstacle slots via MjSpec:** Obstacle body+geom slots are injected programmatically via `SceneComposer.prepare_spec(spec)` before model compilation (uses MuJoCo's MjSpec API). The slot count is configurable (default: 8 objects x 8 geoms). `worlds/room.xml` contains only the static arena (floor, curbs, target, distractors) — no pre-allocated placeholders. Bot `scene.xml` files include `room.xml` via `<include>`.

**Scale progression (planned):** Room → Apartment → House → Village.

## MuJoCo → Visualization Gotchas

1. **Always call `mj_forward()` after `mj_resetData()`**
   Positions/orientations aren't computed until forward kinematics runs

2. **Get meshes from model, not files**
   `model.mesh_vert` and `model.mesh_face` have XML `scale=` already applied

3. **Use model structure to drive everything**
   Iterate `model.nbody`, `model.ngeom`, `model.ncam` - don't hardcode names

4. **Respect geometry hierarchy**
   Bodies → Geoms → Meshes (each has relative transforms)

   ```txt
   world/{body}              (data.xpos, data.xquat)
     └─ {geom}               (model.geom_pos, model.geom_quat)
        └─ mesh              (vertices, faces)
   ```

5. **Cameras need coordinate corrections**
   MuJoCo ≠ Rerun conventions → compose quaternions to fix orientation

6. **Don't duplicate model entities**
   If the model has floor/target/etc., use it - don't log separately

## Key Files

- **main.py** - Entry point + MindSimApp TUI application
- **training/train.py** - Training loop orchestration
- **training/pipeline.py** - Per-bot training config (hyperparameters, rewards)
- **training/run_manager.py** - Run lifecycle, naming, W&B init, run discovery
- **training/checkpoint.py** - Checkpoint save/load/resolve
- **training/policies.py** - Neural networks (LSTMPolicy, TinyPolicy)
- **training/rewards.py** - Reward hierarchy system
- **sim/env.py** - MuJoCo simulation wrapper (SimEnv: step, reset, sensors, camera)
- **sim/scene_preview.py** - Scene gen preview in MuJoCo viewer
- **viz/replay.py** - Download/regenerate Rerun recordings
- **viz/visualize.py** - One-shot Rerun visualization
- **tools/play.py** - Interactive play mode
- **tools/view.py** - MuJoCo viewer
- **scene_gen/** - Procedural scene generation (concepts, composer, primitives)
- **tui/screens/** - All TUI screens (8 screens extracted from main.py)

## Run Management

Each training run gets its own directory under `runs/<run_name>/`:

- **Run naming**: `<bot_abbrev>-<policy>-<MMDD>-<HHMM>` (e.g., `s2w-lstm-0218-1045`). Collision avoidance appends `-2`, `-3`, etc.
- **Bot abbreviations**: `simple2wheeler` -> `s2w`, `simplebiped` -> `biped`, `walker2d` -> `w2d`
- **Metadata**: `run_info.json` tracks status (running/completed/failed), bot, algorithm, W&B link, batch count, etc.
- **W&B**: All runs go to a single `mindsim` project, with tags for bot/algorithm/policy filtering.

## TUI

The interactive TUI (`uv run mjpython main.py`) has a screen-based flow:

```
MainMenuScreen
  ├── [s] Smoketest     -> runs smoketest
  ├── [n] New run       -> BotSelectorScreen -> TrainingDashboard
  ├── [g] Scene gen     -> MuJoCo viewer with procedural scenes (loads room.xml directly)
  ├── [b] Browse runs   -> RunBrowserScreen -> RunActionScreen
  └── [q] Quit
```

**RunActionScreen** actions for a selected run: **[p]** play, **[r]** resume, **[v]** view, **[w]** open W&B.

**Navigation**: `Esc` and `Backspace` consistently go back / leave a screen.

## Training

Current algorithm: **PPO** (Proximal Policy Optimization with GAE)

## Branching & Change Philosophy

1. **Master stays stable.** Only small, safe changes land directly on master.
2. **Experiments are focused.** `exp/YYMMDD-<name>` branches test one hypothesis. No drive-by refactors.
3. **Tooling/infra changes are separate.** `infra/<name>` branches, not mixed into experiments.
4. **Commit logs are a journal.** Explain _why_, not just _what_.

**Before any non-trivial change:** ensure working tree is clean, classify the change, use `make wt-new` for experiments/infra.

**Experiment workflow:** `make wt-new NAME=foo` → work on branch → add entry to `EXPERIMENTS.md` → merge or abandon.

**Tracking:** `EXPERIMENTS.md` records branch, hypothesis, status, outcome, W&B links.

## Future Directions

- **Structural validation (FEA):** Run finite-element analysis on generated meshes to validate printability before fabrication. The immediate smoke test is cross-section analysis (servo stall torque → minimum bridge cross-section at each bracket). The longer-term goal is mesh FEA using `solidspy` or `sfepy` — load each body with servo reaction forces and gravity, check peak stress vs. material yield strength. This would catch thin bridges, under-supported brackets, and other structural failures that the packing solver doesn't reason about.

## Visual Regression Testing

Test renders are committed screenshots of every component, bracket, and bot. Review them in `git diff` after geometry changes to catch regressions visually — they're the parametric CAD equivalent of screenshot tests.

```bash
make renders              # Regenerate everything (~90s)
make renders-components   # Component & bracket tear sheets only
make renders-bots         # Full bot builds only (STEP + STL + XML + renders)
make renders-rom          # Subassembly ROM validation only
```

**After any geometry change**, run `make renders` and review the diffs:

Each stage produces both 3D renders (PNG/PDF) and 2D technical drawings (SVG):

| Stage          | 3D renders                                             | 2D drawings                                                     |
| -------------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| Component      | `botcad/components/test_*.png` — 6-view tear sheets    | `botcad/components/drawing_*.svg` — section views at key planes |
| Bracket        | `botcad/components/test_{bracket,cradle,coupler}*.png` | `botcad/components/drawing_{pocket,coupler,cradle}*.svg`        |
| ROM validation | `botcad/components/test_rom_*.png` — sweep filmstrips  | —                                                               |
| Bot joints     | `bots/*/test_sweep.png` — per-joint ROM filmstrip      | `bots/*/drawings/drawing_joint_*.svg` — bracket-servo sections  |
| Bot assembly   | `bots/*/assembly_visual.pdf`, `test_assembly.pdf`      | —                                                               |
| Bot overview   | `bots/*/test_overview.png`                             | —                                                               |

For interactive debugging during development, use `botcad/debug_drawing.py`:

```python
from botcad.debug_drawing import DebugDrawing
drawing = DebugDrawing("my_debug", scale=15.0)
drawing.add_part("part_a", solid_a)
drawing.add_part("part_b", solid_b)
drawing.add_section("front", Plane.XY.offset(z))
drawing.add_section("side", Plane.XZ)
drawing.save_and_open("debug.svg")  # opens in browser
```

## Rendering Architecture

All visual output flows through two centralized modules:

**3D rendering** (`botcad/emit/render3d.py`):

- `SceneBuilder` — declarative MuJoCo XML construction (replaces hand-rolled f-string XML)
- `Renderer3D` — 3-pass pipeline: color → segmentation → depth, with post-processing (edge detection, SSAO, white background)
- `Color` dataclass — single source for RGBA, derives MuJoCo string and PIL RGB
- Orthographic projection, CAD-style lighting (high ambient, low specular), multisampled
- Standard view presets: `VIEWS_6` (front/back/left/right/top/iso), `VIEWS_4`
- Camera auto-centers on mesh geometry bounds, ignoring debug overlays

**2D compositing** (`botcad/emit/composite.py`):

- `grid()` — N×M view grid with title, color legend, view labels
- `filmstrip()` — horizontal strip with collision indicators and ROM bar
- Font pipeline: Input Sans Narrow Bold/Regular → DejaVu Sans → Arial → default

**Usage pattern**: build a `SceneBuilder`, call `to_xml()`, create `Renderer3D`, call `render_views()` or `render_frame()`, composite with `grid()` or `filmstrip()`.

## Development Notes

- **Run `make validate` after every major step** - This runs lint, tests, and regenerates all renders. Do this after any change to bot designs, skeleton DSL, packing, geometry, brackets, or emitters. Review render diffs before committing. If validation fails, fix the issue before moving on.
- **Clean up before committing** - Remove debug scripts (debug*\*.py, test*\*.py created during dev), temporary files, and .rrd recordings before making commits
- **Bot changes require the checklist** - When creating a new bot or modifying bot XML (geometry, actuators, damping, sensors), read and follow `NEW_BOT_CHECKLIST.md` before committing. Common mistakes it catches: unrealistic joint speeds from wrong damping, gait phase period copied from a different-sized robot, feet not touching the floor, missing contact exclusions.
- **TUI changes require snapshot tests** - After modifying any TUI screen code (`tui/screens/`, `main.py` screen classes), run: `uv run pytest tests/test_tui_snapshots.py -v`. If UI changes are intentional, update baselines with `--snapshot-update`. Snapshot driver apps live in `tests/snapshot_drivers/`.

## Commit Message Format

```
Short summary (imperative mood, <72 chars)

Why this change was made. Context, reasoning, alternatives considered,
observations. The git log is the project's journal.

Session: <link to Claude session>
```
