# MindSim - 2-Wheeler Robot Simulation

Simple 2-wheeler robot with camera for training neural networks in MuJoCo.

## Quick Start

**Always run with `uv`:**

```bash
uv run python <script.py>
```

## Project Structure

```
mindsim/
├── bots/simple2wheeler/
│   ├── bot.xml              # Robot: bodies, joints, cameras, meshes
│   ├── scene.xml            # World: floor, lighting, target
│   └── meshes/*.stl         # Visual geometry (scaled in XML)
├── simple_wheeler_env.py    # Environment API
└── visualize.py             # Rerun visualization → robot_sim.rrd
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
uv run python visualize.py
rerun robot_sim.rrd
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

   ```
   world/{body}              (data.xpos, data.xquat)
     └─ {geom}               (model.geom_pos, model.geom_quat)
        └─ mesh              (vertices, faces)
   ```

5. **Cameras need coordinate corrections**
   MuJoCo ≠ Rerun conventions → compose quaternions to fix orientation

6. **Don't duplicate model entities**
   If the model has floor/target/etc., use it - don't log separately

## Key Files

- **bot.xml** - Robot structure (motors, sensors, camera, meshes)
- **scene.xml** - World setup (target, floor, lighting)
- **simple_wheeler_env.py** - Env logic (step, reset, reward)
- **visualize.py** - Rerun logging (model-driven, no hardcoding)

## Training

Current algorithm: **REINFORCE** (vanilla policy gradient with stochastic policy)

### Future experiments to try

- **Reward-Weighted Regression (RWR)** - Simpler, treat RL as weighted supervised learning
- **PPO** - More stable, better sample efficiency, requires critic network

## Experiment Organization

**Hyperparameter tweaks** → Commits on main + W&B experiments

- Learning rate, batch size, reward coefficients, etc.
- W&B tracks the params; git tracks when/why they changed

**Simple implementation changes / bugfixes** → Commits on main

- Bug fixes, small refactors, code cleanup
- Changes that are clearly improvements, not experiments

**Larger experimental changes** → Branches with `exp/` prefix

- Significant code changes exploring a hypothesis
- Examples: `exp/curriculum-target-distance`, `exp/ppo-baseline`, `exp/reward-shaping-v2`
- Keeps mainline clean; avoids disabled feature-flag complexity

**Tracking experiments**: Maintain `EXPERIMENTS.md` in main with:

- Branch name
- Hypothesis (what you're testing)
- Outcome (worked / didn't / partially / merged to main)
- Link to relevant W&B runs

## Development Notes

- **Clean up before committing** - Remove debug scripts (debug*\*.py, test*\*.py created during dev), temporary files, and .rrd recordings before making commits

## Commit Message Formatit's

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
