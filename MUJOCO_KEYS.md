# MuJoCo Keyboard & Mouse Cheat Sheet

Covers the built-in `simulate` viewer and the `mujoco-python-viewer` library.

## Simulation Control

| Key | Action |
|-----|--------|
| `Space` | Pause / unpause simulation |
| `Right Arrow` | Step forward one timestep (while paused) |
| `Backspace` | Reset simulation |
| `S` | Halve playback speed |
| `F` | Double playback speed |
| `D` | Toggle frame-skipping (render every frame vs. skip when lagging) |
| `Esc` | Quit viewer |

## Camera

| Key | Action |
|-----|--------|
| `Tab` | Cycle through cameras (fixed cameras, then free) |
| `Ctrl+S` | Save current camera config (python viewer) |

## Visualization Toggles

| Key | Action |
|-----|--------|
| `C` | Contact points & forces |
| `J` | Joint axes |
| `E` | Cycle frame visualization modes |
| `R` | Transparent geoms |
| `I` | Inertia ellipsoids |
| `M` | Center-of-mass markers |
| `O` | Shadows on/off |
| `V` | Convex hull rendering |
| `W` | Wireframe mode |
| `0`-`5` | Toggle geom groups 0-5 |

## UI & Recording

| Key | Action |
|-----|--------|
| `F1` | Built-in help overlay (simulate) |
| `H` | Toggle help/overlay menu |
| `G` | Toggle graph overlay |
| `T` | Screenshot (saved to `/tmp/frame_*.png`) |
| `Alt` (hold) | Show menu bar (python viewer) |

## Mouse Controls

### Camera Movement

| Input | Action |
|-------|--------|
| Left drag | Rotate (vertical) |
| Shift + Left drag | Rotate (horizontal) |
| Right drag | Pan (vertical) |
| Shift + Right drag | Pan (horizontal) |
| Scroll wheel | Zoom |

### Object Interaction

| Input | Action |
|-------|--------|
| Left double-click | Select body |
| Right double-click | Set camera focus point |
| Ctrl + Right double-click | Track selected body |
| Ctrl + Left drag | Apply torque to selected body |
| Ctrl + Right drag | Apply force to selected body |

## Tips

- Right-click GUI buttons in `simulate` to see their keyboard shortcut
- `simulate` shortcuts are defined in `simulate.cc` in the MuJoCo source
- The python viewer (`mujoco-python-viewer`) mirrors most of these but adds screenshot/video features
