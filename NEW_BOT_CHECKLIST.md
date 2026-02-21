# New Bot Design Checklist

Hard-won lessons from the duck biped and simple arm. **Read this before building or modifying any bot.** Mistakes here are expensive — you won't discover them until hours into training, when the policy fails to learn or learns something wrong.

---

## Part 1: Universal (All Bots)

These apply to every bot regardless of type.

### 1. Actuator Type — Position vs Torque

| | Position Actuators | Torque Motors |
|---|---|---|
| **ctrl=0 behavior** | Springs hold joints at default angles | Zero force — robot collapses under gravity |
| **Passive stability** | Yes (free standing / holding pose) | No (must actively balance or hold) |
| **RL difficulty** | Policy outputs target angles | Policy must learn physics from scratch |
| **Best for** | First pass, getting behavior to work | Advanced policies, full dynamics control |

**Start with position actuators.** They provide a stable baseline and let the policy focus on the task (walk, grasp) rather than fighting gravity. You can always switch to torque later.

### 2. Motor Gains (kp) — Scale to Body/Joint

Position actuator spring gain (kp) must match the mass being moved and the lever arm:

- **Too high**: joints snap violently to targets, oscillation, jitter
- **Too low**: robot sags, can't hold pose, sluggish response

Guidelines:
- Heavy joints (shoulder, hip): kp = 30-50
- Medium joints (elbow, knee): kp = 20-30
- Light joints (wrist, ankle, fingers): kp = 5-20

**Smaller body = lower kp.** The original biped (12 kg) used kp=100; the duck biped (10 kg, shorter legs) needed kp=40. The arm shoulder (2 kg upper arm) uses kp=50; the gripper (50g fingers) uses kp=20.

### 3. Joint Damping — Prevent Oscillation

Damping resists joint velocity. Too low = underdamped oscillation. Too high = sluggish movement.

The damping ratio ζ = damping / (2 * sqrt(kp * I)), where I is the effective inertia. For RL, aim for ζ ≈ 0.3-0.5 (slightly underdamped — responsive but not wild).

**Rule of thumb**: damping ≈ 5-10% of kp for small robots. Example (arm): shoulder kp=50, damping=2.0; wrist kp=10, damping=0.3.

### 4. Contact Exclusions — Prevent Self-Collision Jitter

Always exclude collisions between adjacent bodies in a kinematic chain. Missing exclusions cause the MuJoCo solver to fight self-intersection, creating jitter and instability that wastes training.

```xml
<contact>
  <exclude body1="parent" body2="child" />
</contact>
```

Walk the full chain and add an exclusion for every parent↔child pair.

### 5. Sensor Sanity — Verify Dimensions Match Config

The `sensor_input_size` in config **must** exactly match the number of sensor values the model produces (plus any synthetic inputs like gait phase). A mismatch silently feeds garbage to the policy.

```bash
# Print actual sensor count from the model
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml')
print(f'sensor_dim = {m.nsensordata}')
for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f'  {name}: dim={m.sensor_dim[i]}')
"
```

### 6. Quick Validation Sequence

After building or modifying a bot:

```bash
# 1. Does it load?
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml'); print('OK')"

# 2. Does it look right? (visual check)
uv run mjpython main.py view --bot BOTNAME

# 3. Does the training pipeline work end-to-end?
uv run mjpython main.py train --bot BOTNAME --smoketest
```

---

## Part 2: Walking Bots (Bipeds)

Additional checks for any bot that needs to stand, balance, or walk.

### 7. Geometry — Feet Must Touch the Floor

Calculate the kinematic chain from base to foot bottom. The base z-position must place foot bottoms at z=0 (within 1-2mm).

```
base_z = hip_offset_z + upper_leg_length + lower_leg_length + foot_offset_z + foot_half_height
```

If feet start above the floor, the robot drops on spawn and the impact destabilizes it. If below, feet penetrate the floor and MuJoCo fights to push them out.

### 8. Foot Shape — Flat Bottoms Required for 3D

- **Box feet**: area contact with floor. Required for lateral stability in 3D.
- **Ellipsoid/sphere feet**: single contact point — zero lateral support. Falls immediately.
- **Capsule feet**: line contact. Slightly better but still inadequate for 3D.
- **Curved feet** (passive walker literature): only work in 2D sagittal-plane models.

**Rule: always use box or composite feet for 3D bipeds.**

### 9. Mass Distribution — Heavy Top, Light Legs

Concentrate mass in the torso (55-80% of total). Lighter legs swing faster and require less motor effort. Matches successful passive walkers.

Use `inertiafromgeom="true"` with `density="1000"` for auto-computed inertia on leg geoms. Use explicit `mass="X"` on the torso to control the ratio.

### 10. Stance Width — Wider = More Laterally Stable

Lateral stability is proportional to `stance_width / CoM_height`. A ratio > 1 provides strong lateral stability but may restrict natural gait.

**Don't extend feet forward to fix fore-aft stability** — this shifts the support polygon center and can make things worse. Ensure CoM projects onto center of foot contact area.

### 11. Stability Testing

Run `stability_test.py` after every geometry change:

```bash
uv run python stability_test.py --scene bots/BOTNAME/scene.xml --verbose
```

Three tests:
1. **Passive Standing** (10s, zero control): Does it stand? Non-negotiable.
2. **Perturbation Resistance** (impulses): How robust to pushes? Target: > 5 N·s lateral, > 2 N·s forward.
3. **Mobility-Stability Tradeoff** (sinusoidal gait): Can legs move without falling? Target: stable at amplitude ≥ 0.05.

### 12. Fall Detection Config

Set these in `EnvConfig` so training terminates fallen episodes early (saves compute):

- `fall_height_fraction`: fraction of initial height below which = fallen (e.g. 0.5)
- `fall_up_z_threshold`: min torso up_z to be "healthy" (e.g. 0.54 = ~57° from vertical)
- `fall_grace_steps`: consecutive unhealthy steps before termination (e.g. 50 at 125Hz = 0.4s)

---

## Part 3: Manipulator / Grasping Bots

Additional checks for any bot that needs to reach, grasp, or manipulate objects.

### 13. Friction — The Silent Training Killer

Friction coefficients determine whether the gripper can hold anything. MuJoCo friction has three components: `friction="slide spin roll"`.

**Finger friction must be high enough to resist gravity on the object.** The grip force equation is roughly: `F_grip * μ ≥ m_object * g`. If μ is too low, the object slides out even with perfect finger placement, and the policy can never learn to lift.

Guidelines:
- **Finger geoms**: `friction="2.0 1.0 0.01"` — high slide friction is critical
- **Object (cup/target)**: `friction="1.5 0.5 0.01"` — moderate, should resist sliding on table too
- **Table surface**: `friction="1.0 0.5 0.01"` — default is fine, object shouldn't slide around on its own

**Test it**: with the gripper closed around the object, can MuJoCo hold it against gravity? If the object slips through the fingers during a `mj_forward` call with fingers at closed-position, friction is too low.

### 14. Contact Dimensionality (condim) — Get the Physics Right

`condim` controls how many contact force directions MuJoCo uses:

| condim | Forces | Use case |
|--------|--------|----------|
| 1 | Normal only (frictionless) | Walls, floors you don't walk on |
| 3 | Normal + 2D tangent friction | **Default.** Good for most grasping |
| 4 | + torsional friction | Round objects that spin in the grip |
| 6 | + rolling friction | Balls, cylinders that roll |

For grasping:
- **Fingers**: `condim="3"` minimum. Use `condim="4"` for cylindrical objects (cups, bottles) to prevent spin.
- **Object**: should match finger condim
- **Table**: `condim="3"` is fine

**Mismatch trap**: if the floor has `condim="3"` but the object has `condim="1"`, the object will slide frictionlessly on the floor even though friction values are set. Both sides of a contact need compatible condim.

### 15. Object Mass vs Grip Force — Can You Actually Hold It?

The gripper must be physically capable of holding the object. Check:

1. **Object mass**: keep it light (10-100g for small grippers). A 50g cup is reasonable.
2. **Gripper kp**: must generate enough force to squeeze. With kp=20 and a slide joint, max force ≈ kp * max_range. For kp=20, range=0.025m → max force ≈ 0.5N. That holds ~50g against gravity (F=mg=0.5N), but barely. Consider raising gripper kp if the object is heavier.
3. **Finger geometry**: fingers must actually enclose the object. Check that closed-finger gap < object diameter < open-finger gap.

```bash
# Verify grip geometry
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
# Check finger range
j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'finger_left')
print(f'finger range: {m.jnt_range[j]}')
print(f'finger closed gap vs object size — eyeball this in viewer')
"
```

### 16. Touch Sensors — Placement and Signal Quality

Touch sensors give the policy contact feedback for grasping. Without them, the policy must infer contact purely from vision or joint torques (much harder).

**Placement**: sites should be on the **inner face** of each finger, where contact with the object occurs. Placing them on the outer face or finger back produces useless signals.

**Signal check**: after a grasp attempt, verify touch sensor values are non-zero when fingers contact the object:

```bash
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    if 'touch' in name:
        print(f'{name}: {d.sensordata[m.sensor_adr[i]]}')
"
```

If touch sensors always read 0 even when fingers are closed on the object, the sites are misplaced or the object's contype/conaffinity don't match.

### 17. Workspace Reachability — Can the Arm Reach the Task?

The arm's kinematic reach must cover all positions where the object might spawn. If any spawn position is unreachable, those episodes are wasted training (the policy literally can't succeed).

Check:
- **Max reach** = sum of all link lengths from shoulder to gripper tip
- **Min reach** = limited by joint range limits (elbow can't straighten past 0°, etc.)
- **Object spawn range** in config must fall within the reachable workspace

For the simple arm: upper_arm=0.25m + forearm=0.20m + wrist/gripper≈0.07m = **0.52m max reach** from shoulder. But shoulder is at z=0.62 and table is at z=0.40, so the arm reaches ~0.22m below shoulder + 0.52m link length in any direction.

### 18. Object Stability — Does It Stay Put Before Grasping?

The object must rest stably on its surface until the arm reaches it. A cup that tips over on spawn wastes training and confuses the reward signal.

**Test**: spawn the object, run 100 timesteps with no actuation. Did it move? If it slides or tips, check:
- Object center of mass vs geometry (is it top-heavy?)
- Table friction (too low?)
- Object initial position (resting on surface, not floating above?)
- Freejoint initial quaternion (upright?)

### 19. Gripper Coupling — Equality Constraints

For two-finger grippers, couple both fingers with an `<equality>` constraint so one actuator controls both symmetrically. This halves the action space and prevents the policy from wasting time learning symmetric finger control.

```xml
<equality>
  <joint joint1="finger_left" joint2="finger_right" polycoef="0 1 0 0 0" />
</equality>
```

**Verify**: actuate the gripper motor in the viewer. Both fingers should open/close symmetrically.

### 20. Control Frequency — Higher for Manipulation

Grasping requires finer control than locomotion. A 10Hz control loop (fine for wheeled robots) is too coarse for manipulation — the gripper overshoots the object.

Guidelines:
- **Wheeled robots**: 10 Hz (mujoco_steps_per_action=5)
- **Walking bots**: 125 Hz (mujoco_steps_per_action=4)
- **Manipulation**: 100-250 Hz (mujoco_steps_per_action=2-5 with timestep=0.002)

Higher frequency = more steps per episode = longer training. Balance precision vs training cost. The simple arm uses 250Hz (timestep=0.002, steps_per_action=2).
