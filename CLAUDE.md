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
make wt-new NAME=my-experiment              # exp/YYMMDD-my-experiment branch
make wt-new NAME=better-tui TYPE=infra      # infra/better-tui branch
make wt-new DESC='implement PPO'            # Claude suggests a branch name
make wt-ls                                  # List all worktrees
make wt-rm NAME=my-experiment               # Remove worktree (keeps branch)
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
├── sim_env.py               # Environment API (SimEnv)
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
from sim_env import SimEnv

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
- **sim_env.py** - MuJoCo simulation wrapper (SimEnv: step, reset, sensors, camera)
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

## Branching & Change Philosophy

### Guiding principles

1. **Master stays stable.** Only small, safe changes land directly on master: hyperparameter tweaks, minor bugfixes, config changes. No big refactors, no risky rewrites.
2. **Experiments are focused.** An `exp/` branch tests a hypothesis — it should not also reorganize the codebase or refactor tooling. Keep the diff reviewable and the intent clear.
3. **Tooling/infra changes are separate.** Improvements to the training pipeline, TUI, visualization, build system, etc. go on `infra/` branches, not mixed into experiments.
4. **Commit logs are a journal.** Every commit should tell a reader *why* the change was made, what was tried, and what was learned. The git log is the project's memory.

### Before ANY non-trivial change

**Always bookmark your starting point before writing code.** You must be able to get back to a known-good state.

1. **Ensure the working tree is clean.** Run `git status` to verify. Commit or stash first.
2. **Classify the change** — decide which category it falls into (see below).

### Change categories

**Hyperparameter tweaks / minor bugfixes** → Commit directly on master

- Learning rate, batch size, reward coefficients, small bugfixes, config changes.
- Working tree must be clean before starting.
- Keep changes small and low-risk.

**Experiments** → `exp/YYMMDD-<name>` branch via worktree

- New algorithms, architecture changes, reward structures, training approaches — anything where the outcome is uncertain.
- **Stay focused**: only make changes that serve the experiment's hypothesis. No drive-by refactors, no structural changes.
- Workflow:
  1. Ensure working tree is clean (committed on master).
  2. Create a worktree: `make wt-new NAME=<descriptive-name>`
     - This creates `../mindsim-<name>/` on branch `exp/YYMMDD-<name>` (date auto-prefixed)
  3. `cd ../mindsim-<name>/ && claude` to start a Claude Code session in the worktree.
  4. Add an entry to `EXPERIMENTS.md` on the branch.
  5. Now begin writing code.
- If the experiment succeeds, merge to master: `git merge exp/YYMMDD-<name>`
- Clean up: `make wt-rm NAME=<name>` (then `git branch -d exp/YYMMDD-<name>` if merged)
- If it fails, remove the worktree — the branch stays as a record.

**Tooling / infrastructure improvements** → `infra/<name>` branch via worktree

- TUI improvements, visualization changes, build system updates, training pipeline refactors, new CLI features.
- Workflow:
  1. Ensure working tree is clean.
  2. Create a worktree: `make wt-new NAME=<name> TYPE=infra`
     - This creates `../mindsim-<name>/` on branch `infra/<name>`
  3. Work, commit, merge back to master when done.
- These can be larger structural changes — that's fine, just keep them separate from experiments.

### Parallel experiments with worktrees

Worktrees let you run multiple experiments simultaneously, each with its own Claude Code session:

```bash
# From the main repo (on master), spin up parallel experiments
make wt-new NAME=ppo-baseline
make wt-new NAME=reward-shaping-v2

# Infra work in parallel
make wt-new NAME=better-tui TYPE=infra

# Each gets its own isolated directory + Claude session
cd ../mindsim-ppo-baseline && claude
cd ../mindsim-reward-shaping-v2 && claude    # (in another terminal)

# Main repo stays clean on master — you can train/view/etc. uninterrupted
make wt-ls
```

- Each worktree is a full working copy with its own branch — no file conflicts between sessions.
- `CLAUDE.md` and all project files are available in each worktree (checked into git).
- Worktrees share the same `.git` database, so branches/history are shared.

### Tracking experiments

Maintain `EXPERIMENTS.md` in main (and on experiment branches) with:

- Branch name (including date prefix)
- Hypothesis (what you're testing and why)
- Status (in-progress / succeeded / failed / partially worked / merged to main)
- Outcome notes (what was learned)
- Link to relevant W&B runs

## Development Notes

- **Clean up before committing** - Remove debug scripts (debug*\*.py, test*\*.py created during dev), temporary files, and .rrd recordings before making commits

## Commit Message Format

The git log is the project's journal. Each commit should be useful to a future reader trying to understand *why* things are the way they are.

```text
Short summary (imperative mood, one line)

What changed and why. Not just "updated X" — explain the motivation, the
problem being solved, or the hypothesis being tested. Include context that
won't be obvious from the diff alone: what alternatives were considered,
what failed first, what tradeoffs were made.

For experiment branches, note observations: did the change help? What
metrics moved? What surprised you?

Session: <link to Claude session>
```

- **First line**: Concise summary (imperative mood, <72 chars)
- **Body**: The "journal entry" — context, reasoning, observations. Use markdown. Be generous with detail; a too-verbose commit message is far better than a cryptic one.
- **Session link**: Link to the Claude Code session (when applicable)
