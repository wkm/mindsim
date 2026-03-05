# 3D Bot Viewer — Implementation Plan

## Architecture

**MuJoCo WASM** for physics/kinematics + **Three.js** for rendering & custom overlays.
Vanilla ES modules from CDN, no build step. Served via `main.py web` command.

```
viewer/
├── index.html              # Single page app, mode tabs
├── viewer.js               # Main entry: init MuJoCo WASM, Three.js scene, mode router
├── mujoco-handler.js       # Load MJCF, step simulation, read body/joint state
├── assembly-mode.js        # Assembly slider, step-by-step animation
├── joint-mode.js           # Joint sliders, angle arcs, axis arrows
├── ik-mode.js              # Anchor selection, drag target, IK solving via MuJoCo
└── ui.js                   # Shared UI: mode tabs, info panel, slider helpers
```

**Python side:**
- `botcad/emit/viewer.py` — Emits `viewer_manifest.json` with structured assembly steps, joint metadata, body hierarchy
- `main.py` gets a `web` subcommand that starts a local HTTP server and opens the browser

## Data Flow

```
bot.xml + meshes/*.stl  ──→  MuJoCo WASM loads MJCF directly
viewer_manifest.json    ──→  JS reads for assembly steps, labels, component metadata
```

MuJoCo WASM gives us:
- Body transforms (positions, orientations) from `mj_data.xpos`, `mj_data.xquat`
- Joint angles from `mj_data.qpos`
- Joint metadata (axis, range, type) from `mj_model`
- FK: set `qpos` → `mj_forward()` → read body transforms
- IK: use `mj_inverse` or equality constraints + stepping

## viewer_manifest.json Structure

```json
{
  "bot_name": "wheeler_arm",
  "bodies": [
    {"name": "base", "mesh": "base.stl", "parent": null},
    {"name": "left_rim", "mesh": "left_rim.stl", "parent": "base", "joint": "left_wheel"}
  ],
  "joints": [
    {
      "name": "shoulder_yaw",
      "parent_body": "base",
      "child_body": "turntable",
      "axis": [0, 0, 1],
      "range": [-1.5, 1.5],
      "pos": [0, 0, 0.04],
      "continuous": false,
      "servo": "STS3215"
    }
  ],
  "assembly_steps": [
    {
      "step": 1,
      "title": "Base body",
      "description": "Print base shell, mount Pi and battery",
      "bodies": ["base"],
      "components": ["pi", "battery"],
      "fasteners": ["mount_base_pi_m1", "mount_base_pi_m2", ...]
    },
    {
      "step": 2,
      "title": "Left wheel",
      "description": "Insert STS3215 servo, attach left_rim with wheel",
      "bodies": ["left_rim"],
      "servos": ["left_wheel"],
      "components": ["wheel"],
      "fasteners": ["screw_left_wheel_ear_1", ...]
    }
  ],
  "ik_chains": [
    {
      "name": "arm",
      "joints": ["shoulder_yaw", "shoulder_pitch", "elbow", "wrist"],
      "end_effector": "hand"
    }
  ]
}
```

## Phase 1: Foundation — MuJoCo WASM + Three.js Scene

**Goal:** Load bot.xml in browser, render all meshes with orbit camera.

1. Create `viewer/` directory with `index.html`
2. Import MuJoCo WASM from CDN (`https://cdn.jsdelivr.net/npm/mujoco-wasm@latest/`)
   - Actually: use the npm package approach — download mujoco_wasm.js + mujoco_wasm.wasm
   - Or use the google-deepmind/mujoco repo's JS examples as reference
3. Import Three.js from CDN (`https://cdn.jsdelivr.net/npm/three@0.170/`)
4. `mujoco-handler.js`: Initialize MuJoCo, load bot.xml + STLs into virtual FS, create model
5. `viewer.js`: Create Three.js scene, sync MuJoCo body transforms to Three.js meshes each frame
6. Orbit controls, lighting, ground grid
7. Add `web` subcommand to `main.py`: `python -m http.server` pointing at bot dir, with viewer files accessible

**Serving strategy:** The web command serves from the bot's output directory (e.g., `bots/wheeler_arm/`), with the viewer JS files symlinked or served from a separate path. Better: serve everything from the project root, with the viewer at `/viewer/` and bot assets at `/bots/{name}/`.

## Phase 2: Joint Validation Mode

**Goal:** Sliders for each joint, visual angle arcs and axis arrows.

1. `joint-mode.js`: Parse joint metadata from MuJoCo model (`mj_model.jnt_range`, `mj_model.jnt_axis`)
2. Generate a slider per joint with range limits, current angle display
3. On slider change: set `mj_data.qpos[joint_idx]` → `mj_forward()` → update Three.js transforms
4. Draw axis arrows: Three.js `ArrowHelper` at each joint position, along joint axis
5. Draw angle arcs: Three.js `Line` geometry showing current angle as an arc sweep
6. Color coding: green = within safe range, yellow = near limit, red = at limit
7. Label each joint with name + current angle in degrees

## Phase 3: Assembly Mode

**Goal:** Step-through assembly with slider, components appearing and inserting.

1. `botcad/emit/viewer.py`: Emit `viewer_manifest.json` from the Bot skeleton
   - Walk the kinematic tree, generate assembly steps in dependency order
   - Each step lists: bodies to show, servos to insert, components to mount, fasteners
2. `assembly-mode.js`: Read manifest, create main slider (0 to N steps)
3. For each step transition:
   - **Body appear:** Fade in mesh from transparent, animate from offset position to final position
   - **Servo insert:** Show green servo box, animate it sliding into position along joint axis
   - **Fasteners:** Show screw cylinders appearing at positions
   - **Components:** Show Pi/battery/camera boxes appearing at mount positions
4. MuJoCo geom groups help here: group 0 = structural meshes, group 1 = servos/fasteners/components, group 2 = wires
5. Use MuJoCo's `mj_data.xpos` for body positions (all transforms are correct from the physics model)
6. Sub-steps within each assembly step (continuous slider or play button with animation)

## Phase 4: IK Mode

**Goal:** Click to anchor a body, drag another body, joints flex to follow.

1. `ik-mode.js`: Raycaster to select bodies on click
2. First click = anchor (highlight in blue), second click = drag target (highlight in red)
3. On mouse drag: compute world-space target position from mouse ray
4. IK solving approach using MuJoCo:
   - Create an equality constraint (weld) at the target body to the mouse position
   - Step the simulation — MuJoCo's solver will move joints to satisfy the constraint
   - Alternative: Jacobian-based IK using `mj_jac` to get the Jacobian, then solve iteratively
5. Highlight the active kinematic chain
6. Show joint angles updating in real-time as the user drags
7. Respect joint limits (MuJoCo handles this natively)

## Phase 5: `main.py web` Command

1. Add `web` subcommand to argparse in `main.py`
2. Start `http.server` on port 8080 (configurable)
3. Serve project root so `/viewer/index.html` and `/bots/wheeler_arm/` are both accessible
4. Auto-open browser to `http://localhost:8080/viewer/?bot=wheeler_arm`
5. Pass bot name as URL parameter

## Implementation Order

1. **Phase 1** — Get something rendering in the browser (most critical, proves the approach)
2. **Phase 2** — Joint validation (simplest interactive mode, immediate utility)
3. **Phase 5** — Serving command (makes it easy to use)
4. **Phase 3** — Assembly mode (needs manifest emitter, more animation work)
5. **Phase 4** — IK mode (most complex, needs equality constraints or Jacobian solver)

## Open Questions / Risks

- **MuJoCo WASM CDN availability:** The `mujoco-wasm` npm package exists but may need specific loading. Fallback: vendor the WASM files in `viewer/vendor/`.
- **STL loading:** MuJoCo WASM needs STL files in its virtual filesystem. We'll need to fetch them and write to MEMFS before loading the model.
- **Scene.xml vs bot.xml:** We should load `bot.xml` directly (not `scene.xml` which includes room.xml). For the viewer we want just the robot, not the arena.
- **Performance:** STL meshes are small (generated per-body). Should be fine for real-time rendering.
