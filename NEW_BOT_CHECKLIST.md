# New Bot Design Checklist

Hard-won lessons from the duck biped and simple arm. **Read this before building or modifying any bot.** Mistakes here are expensive — you won't discover them until hours into training, when the policy fails to learn or learns something wrong.

---

## Joints & Actuators

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
- Light joints (wrist, ankle): kp = 5-20
- Gripper fingers: kp must generate enough force to hold the object (see "Object Mass vs Grip Force")

**Smaller body = lower kp.** The original biped (12 kg) used kp=100; the duck biped (10 kg, shorter legs) needed kp=40. The arm shoulder (2 kg upper arm) uses kp=50.

### 3. Joint Damping — Prevent Oscillation

Damping resists joint velocity. Too low = underdamped oscillation. Too high = sluggish movement.

The damping ratio ζ = damping / (2 * sqrt(kp * I)), where I is the effective inertia. For RL, aim for ζ ≈ 0.3-0.5 (slightly underdamped — responsive but not wild).

**Rule of thumb**: damping ≈ 5-10% of kp for small robots. Example (arm): shoulder kp=50, damping=2.0; wrist kp=10, damping=0.3.

### 4. Joint Ranges — Match the Task

Joint limits (`range` attribute) must allow the robot to complete its task. An arm that can't reach the table, or legs that can't swing enough to walk, will waste training.

Verify by manually setting joints to extreme positions in the viewer and checking that all target positions are reachable.

---

## Contacts & Friction

### 5. Contact Exclusions — Prevent Self-Collision Jitter

Always exclude collisions between adjacent bodies in a kinematic chain. Missing exclusions cause the MuJoCo solver to fight self-intersection, creating jitter and instability that wastes training.

```xml
<contact>
  <exclude body1="parent" body2="child" />
</contact>
```

Walk the full chain and add an exclusion for every parent↔child pair.

### 6. Contact Dimensionality (condim)

`condim` controls how many contact force directions MuJoCo uses:

| condim | Forces | Use case |
|--------|--------|----------|
| 1 | Normal only (frictionless) | Visual-only geoms, walls |
| 3 | Normal + 2D tangent friction | **Default.** Floors, tables, most grasping |
| 4 | + torsional friction | Round objects that spin in a grip |
| 6 | + rolling friction | Balls, cylinders that roll on surfaces |

**Mismatch trap**: if the floor has `condim="3"` but the object has `condim="1"`, the object slides frictionlessly on the floor even though friction values are set. Both sides of a contact need compatible condim.

### 7. Friction Coefficients

MuJoCo friction has three components: `friction="slide spin roll"`.

The effective friction between two geoms is the element-wise product of their friction vectors. So finger `friction="2.0"` against cup `friction="1.5"` gives effective slide friction = 2.0 * 1.5 = 3.0 (MuJoCo actually uses `max(f1, f2)` for each component by default — check `condim` docs).

Guidelines:
- **Floor**: `friction="1.0 0.5 0.01"` — default is fine
- **Table surface**: `friction="1.0 0.5 0.01"` — objects shouldn't slide around
- **Gripper fingers**: `friction="2.0 1.0 0.01"` — high slide friction is critical for holding objects
- **Graspable objects**: `friction="1.5 0.5 0.01"` — moderate, must resist sliding on table

**Test it**: with the gripper closed around the object, can MuJoCo hold it against gravity? If the object slips through during simulation, friction is too low.

### 8. Contact Stiffness — Prevent Sinking

MuJoCo's default contact solver is slightly soft — objects sink into surfaces by 1-2mm. For manipulation, this matters because it shifts the object away from its expected spawn position.

Stiffen table/surface contacts with `solref` (time constant, damping ratio) and `solimp` (impedance range):

```xml
<geom name="table_surface" ... solref="0.005 1" solimp="0.999 0.9999 0.001 0.5 2" />
```

This reduces settling from ~2mm to ~0.5mm. Don't over-stiffen (can cause instability with large timesteps).

---

## Sensors

### 9. Sensor Dimensions — Must Match Config

The `sensor_input_size` in config **must** exactly match the number of sensor values the model produces (plus any synthetic inputs like gait phase). A mismatch silently feeds garbage to the policy.

```bash
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml')
print(f'sensor_dim = {m.nsensordata}')
for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f'  {name}: dim={m.sensor_dim[i]}')
"
```

### 10. Touch Sensors — Placement and Signal Quality

Touch sensors give the policy contact feedback for grasping. Without them, the policy must infer contact purely from vision or joint torques (much harder).

**Placement**: sites must be on the **inner face** of each finger, where contact with the object occurs. Placing them on the outer face or finger back produces useless signals.

**Signal check**: actuate the gripper closed on the object, verify touch sensors read non-zero. If they're always 0, the sites are misplaced or the object's `contype`/`conaffinity` don't match the finger geoms.

---

## Geometry & Mass

### 11. Kinematic Chain — Bodies Must Start Where They Should

Calculate the kinematic chain to verify spawn positions:

- **Bipeds**: feet must touch the floor at z=0 (within 1-2mm). Too high = drop on spawn. Too low = floor penetration jitter.
- **Arms**: shoulder height must be above the workspace. Gripper at rest should be above the table.
- **Objects**: must rest on their surface. A cup on a table should have `z = table_top + object_half_height`.

### 12. Mass Distribution

Mass placement affects dynamics profoundly:

- **Bipeds**: heavy top (55-80% in torso), light legs. Lighter legs swing faster, need less motor effort.
- **Arms**: heavier base/shoulder, lighter distal links. Reduces inertia at the end effector.
- **Objects**: keep graspable objects light (10-100g for small grippers). A 50g cup is reasonable.

Use `inertiafromgeom="true"` with `density="1000"` for auto-computed inertia, or explicit `mass="X"` and `fullinertia` for precise control.

---

## Grasping

These apply whenever the bot needs to pick up, hold, or manipulate objects.

### 13. Object Mass vs Grip Force — Can You Actually Hold It?

The gripper must be physically capable of holding the object. The math is simple:

```
max_grip_force = gripper_kp * finger_joint_range
required_force = object_mass * 9.81 / friction_coefficient
safety_margin = max_grip_force / required_force  (want ≥ 2x)
```

Example: gripper kp=40, finger range=0.025m → max force=1.0N. Cup 50g → weight=0.49N. Margin=2.0x. Adequate.

With kp=20, margin was only 1.0x — barely holds the cup, any perturbation causes a drop. **Always verify ≥ 2x margin.**

### 14. Finger Geometry — Encloses the Object

Fingers must physically surround the object when closed:
- **Closed gap** < object diameter (fingers contact the object)
- **Open gap** > object diameter (can position around it)
- **Finger length** covers enough of the object to produce stable grip

Verify in the viewer by actuating the gripper motor.

### 15. Gripper Coupling — Equality Constraints

For two-finger grippers, couple both fingers with an `<equality>` constraint so one actuator controls both symmetrically. This halves the action space and prevents the policy from wasting time learning symmetric finger control.

```xml
<equality>
  <joint joint1="finger_left" joint2="finger_right" polycoef="0 1 0 0 0" />
</equality>
```

### 16. Object Stability — Does It Stay Put?

The object must rest stably on its surface until the arm reaches it. A cup that tips over or slides on spawn wastes training and confuses the reward signal.

**Test**: spawn the object, run 1000 timesteps with no actuation. Measure drift. Target: < 1mm total displacement.

Check if it fails:
- Center of mass vs geometry (top-heavy?)
- Surface friction (too low?)
- Spawn position (floating above surface? use exact `z = surface_top + half_height`)
- Contact stiffness (object sinking into surface? see "Contact Stiffness")
- Freejoint initial quaternion (upright?)

### 17. Workspace Reachability

The arm's kinematic reach must cover all positions where the object might spawn. Unreachable spawn positions are wasted training episodes.

Check: `max_reach = sum(link_lengths)` from shoulder to gripper. Then verify the object spawn range in config falls within this, accounting for joint limits.

---

## Bipeds & Walking

These apply whenever the bot needs to stand, balance, or walk.

### 18. Foot Shape — Flat Bottoms for 3D

- **Box feet**: area contact with floor. Required for lateral stability in 3D.
- **Ellipsoid/sphere feet**: single contact point — zero lateral support. Falls immediately.
- **Capsule feet**: line contact. Still inadequate for 3D.

**Rule: always use box or composite feet for 3D bipeds.**

### 19. Stance Width

Lateral stability is proportional to `stance_width / CoM_height`. A ratio > 1 provides strong lateral stability.

**Don't extend feet forward to fix fore-aft stability** — this shifts the support polygon center and can make things worse. Ensure CoM projects onto center of foot contact area.

### 20. Stability Testing

Run `stability_test.py` after every geometry change:

```bash
uv run python stability_test.py --scene bots/BOTNAME/scene.xml --verbose
```

Three tests:
1. **Passive Standing** (10s, zero control): Does it stand? Non-negotiable.
2. **Perturbation Resistance** (impulses): Target: > 5 N·s lateral, > 2 N·s forward.
3. **Mobility-Stability Tradeoff** (sinusoidal gait): Target: stable at amplitude ≥ 0.05.

### 21. Fall Detection Config

Set these in `EnvConfig` so training terminates fallen episodes early (saves compute):

- `fall_height_fraction`: fraction of initial height below which = fallen (e.g. 0.5)
- `fall_up_z_threshold`: min torso up_z to be "healthy" (e.g. 0.54 ≈ 57° from vertical)
- `fall_grace_steps`: consecutive unhealthy steps before termination (e.g. 50 at 125Hz = 0.4s)

---

## Control & Training

### 22. Control Frequency — Match the Task

Different tasks need different control rates. Too slow = overshoots targets. Too fast = more steps per episode = slower training.

| Task | Frequency | Typical settings |
|------|-----------|-----------------|
| Wheeled robots | 10 Hz | timestep=0.01, steps_per_action=5 |
| Walking | 125 Hz | timestep=0.002, steps_per_action=4 |
| Manipulation | 100-250 Hz | timestep=0.002, steps_per_action=2-5 |

---

## Validation

### 23. Quick Validation Sequence

After building or modifying any bot:

```bash
# 1. Does it load?
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml'); print('OK')"

# 2. Does it look right? (visual check)
uv run mjpython main.py view --bot BOTNAME

# 3. Does the training pipeline work end-to-end?
uv run mjpython main.py train --bot BOTNAME --smoketest
```

For bipeds, also run stability test (item 20). For grasping bots, also run the grip force and object stability checks (items 13, 16).
