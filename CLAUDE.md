# MindSim - 2-Wheeler Robot Simulation

Simple 2-wheeler robot with camera for training neural networks in MuJoCo.

## Quick Start

**Single entry point — always use `main.py` via `mjpython`:**

```bash
uv run mjpython main.py                    # Interactive TUI (default)
uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython main.py play [CHECKPOINT] [--bot NAME] [--run RUN_NAME]  # Play trained policy
uv run mjpython main.py train [--smoketest] [--bot NAME] [--resume REF] [--num-workers N]
uv run mjpython main.py smoketest          # Alias for train --smoketest
uv run mjpython main.py quicksim           # Rerun debug vis
uv run mjpython main.py visualize [--bot NAME] [--steps N]
```

`--bot NAME` accepts a bot directory name (e.g., `simplebiped`, `simple2wheeler`). Default: `simple2wheeler`.

`--run RUN_NAME` resolves the latest checkpoint from a specific run directory (e.g., `--run s2w-lstm-0218-1045`).

Or use Make shortcuts: `make`, `make view`, `make play`, `make train`, `make smoketest`.

**Worktree management:**

```bash
make wt-new DESC='implement PPO'        # Claude suggests a branch name
make wt-new NAME=my-experiment          # Use explicit name
make wt-ls                              # List all worktrees
make wt-rm NAME=my-experiment           # Remove worktree (keeps branch)
```

## Project Structure

```txt
mindsim/
├── main.py                   # Single entry point for all modes
├── run_manager.py            # Run directory lifecycle & W&B init
├── bots/simple2wheeler/
│   ├── bot.xml              # Robot: bodies, joints, cameras, meshes
│   ├── scene.xml            # World: floor, lighting, target
│   └── meshes/*.stl         # Visual geometry (scaled in XML)
├── simple_wheeler_env.py    # Environment API
├── train.py                 # Training loop & policy networks
├── checkpoint.py            # Checkpoint save/load/resolve
├── view.py                  # MuJoCo viewer
├── play.py                  # Interactive play mode
├── visualize.py             # Rerun visualization
└── runs/                    # Per-run directories (gitignored)
    └── s2w-lstm-0218-1045/
        ├── run_info.json    # Run metadata
        ├── checkpoints/     # Model checkpoints
        └── recordings/      # Rerun .rrd files
```

## Environment API

```python
from simple_wheeler_env import SimpleWheelerEnv

env = SimpleWheelerEnv(render_width=128, render_height=128)

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

- **main.py** - Single entry point for all modes (TUI, view, play, train, etc.)
- **run_manager.py** - Run directory lifecycle, run naming, W&B init, run discovery
- **checkpoint.py** - Checkpoint save/load/resolve (searches `runs/` then legacy `checkpoints/`)
- **config.py** - Centralized training configuration (all hyperparameters)
- **bot.xml** - Robot structure (motors, sensors, camera, meshes)
- **scene.xml** - World setup (target, floor, lighting)
- **simple_wheeler_env.py** - Env logic (step, reset, reward)
- **view.py** - MuJoCo viewer (called via `main.py view`)
- **play.py** - Interactive play mode (called via `main.py play`)
- **train.py** - Training loop and policy networks (called via `main.py train`)
- **rerun_wandb.py** - Rerun-W&B integration for eval episode recordings
- **visualize.py** - Rerun visualization (called via `main.py visualize`)

## Run Management

Each training run gets its own directory under `runs/<run_name>/`:

- **Run naming**: `<bot_abbrev>-<policy>-<MMDD>-<HHMM>` (e.g., `s2w-lstm-0218-1045`). Collision avoidance appends `-2`, `-3`, etc.
- **Bot abbreviations**: `simple2wheeler` -> `s2w`, `simplebiped` -> `biped`, `walker2d` -> `w2d`
- **Metadata**: `run_info.json` tracks status (running/completed/failed), bot, algorithm, W&B link, batch count, etc.
- **W&B**: All runs go to a single `mindsim` project, with tags for bot/algorithm/policy filtering.
- **Backward compat**: `resolve_resume_ref("latest")` searches `runs/*/checkpoints/` first, then legacy `checkpoints/`. Old checkpoints keep working.

## TUI

The interactive TUI (`uv run mjpython main.py`) has a screen-based flow:

```
MainMenuScreen
  ├── [s] Smoketest     -> runs smoketest
  ├── [n] New run       -> BotSelectorScreen -> TrainingDashboard
  ├── [b] Browse runs   -> RunBrowserScreen -> RunActionScreen
  └── [q] Quit
```

**RunActionScreen** actions for a selected run: **[p]** play, **[r]** resume, **[v]** view, **[w]** open W&B.

**Navigation**: `Esc` and `Backspace` consistently go back / leave a screen.

## Training

Current algorithm: **PPO** (Proximal Policy Optimization with GAE)

## Experiment Organization

### Before ANY non-trivial change

**Always bookmark your starting point before writing code.** This is the most important rule. You must be able to get back to a known-good state.

1. **Ensure the working tree is clean.** All current work must be committed. Run `git status` to verify. If there are uncommitted changes, commit or stash them first.
2. **Classify the change** — decide which category it falls into (see below) and follow the corresponding workflow.

### Change categories

**Hyperparameter tweaks** → Commit directly on main

- Learning rate, batch size, reward coefficients, etc.
- Working tree must be clean before starting (rule above).
- W&B tracks the run params; git tracks when/why they changed.

**Simple implementation changes / bugfixes** → Commit directly on main

- Bug fixes, small refactors, code cleanup.
- Changes that are clearly improvements, not experiments.
- Working tree must be clean before starting (rule above).

**Larger experimental changes** → Create a worktree BEFORE writing any code

- This includes: new algorithms, significant architecture changes, new reward structures, new training approaches, or anything where the outcome is uncertain.
- Workflow:
  1. Ensure working tree is clean (committed to main).
  2. Create a worktree: `make wt-new NAME=<descriptive-name>`
     - This creates `../mindsim-<name>/` on branch `exp/<name>`
  3. `cd ../mindsim-<name>/ && claude` to start a Claude Code session in the worktree.
  4. Add an entry to `EXPERIMENTS.md` on the branch (see tracking below).
  5. Now begin writing code.
- Branch naming examples: `exp/curriculum-target-distance`, `exp/ppo-baseline`, `exp/reward-shaping-v2`
- If the experiment succeeds, merge to main from the main repo: `git merge exp/<name>`
- Clean up: `make wt-rm NAME=<name>` (then `git branch -d exp/<name>` if merged)
- If it fails, remove the worktree — the branch stays as a record and main is untouched.

### Parallel experiments with worktrees

Worktrees let you run multiple experiments simultaneously, each with its own Claude Code session:

```bash
# From the main repo (on master), spin up parallel experiments
make wt-new NAME=ppo-baseline
make wt-new NAME=reward-shaping-v2

# Each gets its own isolated directory + Claude session
cd ../mindsim-ppo-baseline && claude
cd ../mindsim-reward-shaping-v2 && claude    # (in another terminal)

# Main repo stays clean on master — you can train/view/etc. uninterrupted
# List active worktrees anytime
make wt-ls
```

- Each worktree is a full working copy with its own branch — no file conflicts between sessions.
- `CLAUDE.md` and all project files are available in each worktree (checked into git).
- Worktrees share the same `.git` database, so branches/history are shared.

### Tracking experiments

Maintain `EXPERIMENTS.md` in main (and on experiment branches) with:

- Branch name
- Hypothesis (what you're testing and why)
- Status (in-progress / succeeded / failed / partially worked / merged to main)
- Outcome notes (what was learned)
- Link to relevant W&B runs

## Development Notes

- **Clean up before committing** - Remove debug scripts (debug*\*.py, test*\*.py created during dev), temporary files, and .rrd recordings before making commits

## Commit Message Format

Commits should follow this structure:

```text
Short summary (one sentence)

Detailed description of the changes in 1-2 paragraphs using markdown.
Explain the motivation, what was changed, and any important decisions made.

## Session Summary

Brief summary of what was discussed and decided during the Claude Code session.

Session: <link to Claude session>
```

- **First line**: Concise summary of what changed (imperative mood)
- **Body**: More verbose explanation (~1-2 paragraphs) with context and reasoning
- **Session section**: Summary of the development session conversation + link to the Claude session
