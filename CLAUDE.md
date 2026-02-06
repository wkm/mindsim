# MindSim - 2-Wheeler Robot Simulation

Simple 2-wheeler robot with camera for training neural networks in MuJoCo.

## Quick Start

**This project uses `uv` for package management. Always run Python with:**
```bash
uv run python <script.py>
```

## Project Structure

```
mindsim/
├── bots/simple2wheeler/     # Robot MJCF files + STL meshes
│   ├── bot.xml              # Robot definition (scaled, camera, motors)
│   ├── scene.xml            # Scene with floor + orange target cube
│   └── meshes/              # STL files (body, wheels, camera)
├── simple_wheeler_env.py    # Core environment class
├── test_manual_control.py   # Basic testing script
├── visualize_with_rerun.py  # Rerun visualization (saves .rrd file)
└── visualize_matplotlib.py  # Live matplotlib plots
```

## Environment API

```python
from simple_wheeler_env import SimpleWheelerEnv

env = SimpleWheelerEnv(render_width=16, render_height=16)

# Step simulation with motor commands
camera_img = env.step(left_motor=0.5, right_motor=0.5)  # range: [-1, 1]

# Get state
bot_pos = env.get_bot_position()          # [x, y, z] in meters
target_pos = env.get_target_position()    # [x, y, z] in meters
distance = env.get_distance_to_target()   # float

env.close()
```

## Running Visualizations

**Rerun (recommended):**
```bash
uv run python visualize_with_rerun.py
# Creates robot_sim.rrd file
# View with: rerun robot_sim.rrd (requires: cargo binstall rerun-cli)
```

**Matplotlib (live plots):**
```bash
uv run python visualize_matplotlib.py
# Opens interactive window with camera, motors, distance, trajectory
```

**Basic test:**
```bash
uv run python test_manual_control.py
```

## Scene Configuration

- **Bot**: Starts at origin (0, 0, 0), 90° rotated
- **Target**: Orange cube at (0, 2, 0.08) - 2m in front
- **Camera**: 64x64 RGB (configurable), mounted on bot
- **Motors**: Gear ratio = 10, control range [-1, 1]
- **Distance**: Bot moves ~0.08m per 100 steps at motor=0.5

## Next Steps for ML

1. **Reward function**: `reward = -distance_to_target`
2. **Observation**: 16x16 camera image (grayscale or RGB)
3. **Action**: (left_motor, right_motor) in [-1, 1]
4. **Network**: Simple CNN → 2 output values
5. **Training**: Use wandb (already in dependencies)

## Key Files to Edit

- `bot.xml` - Robot config (motor gear, camera position)
- `scene.xml` - Target position, lighting, floor size
- `simple_wheeler_env.py` - Add reward function, reset logic
