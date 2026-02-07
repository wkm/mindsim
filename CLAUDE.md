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

## Development Notes

- **Clean up before committing** - Remove debug scripts (debug*\*.py, test*\*.py created during dev), temporary files, and .rrd recordings before making commits
