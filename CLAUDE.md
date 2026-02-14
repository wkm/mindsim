# MindSim - 2-Wheeler Robot Simulation

Simple 2-wheeler robot with camera for training neural networks in MuJoCo.

## Quick Start

**Single entry point — always use `tui.py` via `mjpython`:**

```bash
uv run mjpython tui.py                    # Interactive TUI (default)
uv run mjpython tui.py view [--bot NAME]  # MuJoCo viewer
uv run mjpython tui.py play [CHECKPOINT] [--bot NAME]  # Play trained policy
uv run mjpython tui.py train [--smoketest] [--bot NAME] [--resume REF] [--num-workers N]
uv run mjpython tui.py smoketest          # Alias for train --smoketest
uv run mjpython tui.py quicksim           # Rerun debug vis
uv run mjpython tui.py visualize [--bot NAME] [--steps N]
```

`--bot NAME` accepts a bot directory name (e.g., `simplebiped`, `simple2wheeler`). Default: `simple2wheeler`.

Or use Make shortcuts: `make`, `make view`, `make play`, `make train`, `make smoketest`.

## Project Structure

```txt
mindsim/
├── tui.py                   # Single entry point for all modes
├── bots/simple2wheeler/
│   ├── bot.xml              # Robot: bodies, joints, cameras, meshes
│   ├── scene.xml            # World: floor, lighting, target
│   └── meshes/*.stl         # Visual geometry (scaled in XML)
├── simple_wheeler_env.py    # Environment API
├── train.py                 # Training loop & policy networks
├── view.py                  # MuJoCo viewer
├── play.py                  # Interactive play mode
└── visualize.py             # Rerun visualization
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
uv run mjpython tui.py visualize
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

- **tui.py** - Single entry point for all modes (TUI, view, play, train, etc.)
- **bot.xml** - Robot structure (motors, sensors, camera, meshes)
- **scene.xml** - World setup (target, floor, lighting)
- **simple_wheeler_env.py** - Env logic (step, reset, reward)
- **view.py** - MuJoCo viewer (called via `tui.py view`)
- **play.py** - Interactive play mode (called via `tui.py play`)
- **train.py** - Training loop and policy networks (called via `tui.py train`)
- **visualize.py** - Rerun visualization (called via `tui.py visualize`)

## Training

Current algorithm: **REINFORCE** (vanilla policy gradient with stochastic policy)

### Future experiments to try

- **Reward-Weighted Regression (RWR)** - Simpler, treat RL as weighted supervised learning
- **PPO** - More stable, better sample efficiency, requires critic network

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

**Larger experimental changes** → Create an `exp/` branch BEFORE writing any code

- This includes: new algorithms, significant architecture changes, new reward structures, new training approaches, or anything where the outcome is uncertain.
- Workflow:
  1. Ensure working tree is clean (committed to main).
  2. Create and switch to the experiment branch: `git checkout -b exp/<descriptive-name>`
  3. Add an entry to `EXPERIMENTS.md` on the branch (see tracking below).
  4. Now begin writing code.
- Branch naming examples: `exp/curriculum-target-distance`, `exp/ppo-baseline`, `exp/reward-shaping-v2`
- If the experiment succeeds, merge to main. If it fails, the branch stays as a record and main is untouched.

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
