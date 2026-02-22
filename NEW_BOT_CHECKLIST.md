# New Bot Design Checklist

Lessons learned from designing the duck biped. Use this when creating or modifying any walking bot.

## 1. Geometry — Feet Must Touch the Floor

Calculate the kinematic chain from base to foot bottom. The base z-position must place the foot bottoms at z=0 (or within 1-2mm).

```
base_z = hip_offset_z + upper_leg_length + lower_leg_length + foot_offset_z + foot_half_height
```

Example (duck biped):
```
0.22 = 0.05 + 0.08 + 0.06 + 0.015 + 0.015
```

If the feet start above the floor, the robot drops on spawn and the impact can destabilize it. If below, the feet penetrate the floor and MuJoCo's solver fights to push them out.

## 2. Foot Shape — Flat Bottoms Required for 3D

- **Box feet**: provide area contact with the floor. Required for lateral stability in 3D.
- **Ellipsoid/sphere feet**: only produce a single contact point on a flat plane. The robot balances on two points (a line) with zero lateral support polygon. Will fall immediately.
- **Capsule feet**: produce a line contact. Slightly better than ellipsoids but still inadequate for 3D.
- **Curved/rounded feet** (passive walker literature): only work in 2D sagittal-plane models. Do not use for 3D bipeds.

Rule: **always use box or composite feet for 3D bipeds.**

## 3. Actuator Type — Position vs Torque

| | Position Actuators | Torque Motors |
|---|---|---|
| **ctrl=0 behavior** | Springs hold joints at default angles | Zero force — robot collapses under gravity |
| **Passive stability** | Yes (free standing) | No (must actively balance) |
| **RL difficulty** | Policy outputs target angles | Policy must learn balance AND locomotion |
| **Best for** | First pass, getting walking to work | Advanced policies, full dynamics control |

**Start with position actuators.** The gravitational torque on joints (e.g., ~23 Nm at the hip for a 10 kg torso) vastly exceeds what joint damping alone provides (~0.3 Nm). Without position springs, the robot instantly collapses.

## 4. Motor Gains (kp) — Scale to Gravitational Torque

Position actuator spring gain (kp) must resist gravitational torque at each joint. The required kp scales with the mass above the joint and the lever arm to the center of mass.

- **Too high**: joints snap violently to targets, creating oscillation and flailing
- **Too low**: robot sags, can't hold standing pose, sluggish response

kp must be high enough to hold the standing pose, but the resulting movement speed is governed by damping (see section 5). Don't try to control speed by lowering kp below the gravity threshold — use damping instead.

| Robot | Mass | CoM height | Hip kp | Knee kp |
|-------|-----:|-----------:|-------:|--------:|
| Duck biped | 18 kg | 0.22m | 40 | 40 |
| Child biped | 17 kg | 0.52m | 90 | 90 |

The child needs ~2x the duck's kp despite similar mass, because the longer lever arms produce higher gravitational torques. **kp scales with torque demands, not just body size.**

## 5. Joint Damping — Limit Speed, Not Just Oscillation

Damping resists joint velocity. It serves two purposes:
1. **Prevent oscillation** around the target position
2. **Limit joint speed** to physically realistic values

The terminal angular velocity of a position-actuated joint is:

```
terminal_vel = kp * position_error / damping
```

The **tip speed** at the end of the limb is `terminal_vel * lever_arm`. This is what determines whether movement looks realistic or comically fast.

**Critical insight**: damping that works for a short-legged robot will produce absurd speeds on a taller one, even with identical kp. A 3x longer leg at the same angular velocity produces 3x the tip speed.

Practical approach — work backwards from realistic tip speeds:

```
damping = kp * max_position_error / target_angular_vel
```

Reference angular velocities for walking:
- Hip: 4-5 rad/s peak (swing phase)
- Knee: 6-8 rad/s peak (swing phase)
- Ankle: 6-8 rad/s peak

| Robot | Hip kp | Hip damping | Hip terminal vel | Leg tip speed |
|-------|-------:|----------:|----------------:|--------------:|
| Duck (0.17m legs) | 40 | 3.0 | 13.3 rad/s | 2.3 m/s |
| Child (0.46m legs) | 90 | 18.0 | 5.0 rad/s | 2.3 m/s |

The duck gets away with low damping because its legs are short. The child biped needs ~6x more damping to achieve the same tip speed.

**Don't use a fixed damping-to-kp ratio.** Instead, compute terminal velocities and verify tip speeds are physically plausible (< 3 m/s for walking, < 5 m/s for running).

## 6. Mass Distribution — Heavy Top, Light Legs

Concentrate mass in the torso:
- Torso should be 55-80% of total mass
- Lighter legs swing faster, require less motor effort
- Matches successful passive walkers (heavy hip mass)

Use `inertiafromgeom="true"` with `density="1000"` for auto-computed inertia on leg geoms. Use explicit `mass="X"` on the torso geom to control the mass ratio.

## 7. Stance Width — Wider = More Laterally Stable

The lateral stability margin is proportional to `stance_width / CoM_height`.

- Duck biped: stance = 0.28m, CoM height ≈ 0.22m, ratio = 1.27 → very stable laterally (18 N·s impulse resistance)
- The lateral vs forward asymmetry can be extreme: 18 N·s lateral vs 3 N·s forward

**Don't extend feet forward to fix fore-aft stability** — this shifts the support polygon center and can make things worse. Instead, ensure the CoM projects onto the center of the foot contact area.

## 8. Stability Testing

Run `stability_test.py` after every geometry change. Three tests:

1. **Passive Standing** (10s, zero control): Does it stand? Minimum bar for any bot.
2. **Perturbation Resistance** (lateral/forward impulses): How robust to pushes?
3. **Mobility-Stability Tradeoff** (sinusoidal gait at increasing amplitude): Can the legs move without falling?

Target scores:
- Standing: PASS (non-negotiable)
- Impulse: > 5 N·s lateral, > 2 N·s forward
- Mobility: stable at amplitude ≥ 0.05

## 9. Contact Exclusions

Always exclude collisions between:
- Torso ↔ upper legs (they overlap at the hip joint)
- Upper leg ↔ lower leg (adjacent in chain)
- Lower leg ↔ foot (adjacent in chain)
- Left upper leg ↔ right upper leg (can clip through each other)

Missing exclusions cause the solver to fight self-intersection, creating jitter and instability.

## 10. Actuator ctrlrange — Must Match Joint Range

The policy outputs values in `ctrlrange` (default `[-1, 1]`). For position actuators, these values are **target angles in radians**. If ctrlrange doesn't match the joint's `range`, the NN wastes control authority or can't reach valid positions.

Common failures:
- **ctrlrange wider than joint range**: Most of the NN's output space commands impossible positions. The PD controller saturates pushing against the joint stop, wasting force and confusing gradient signals.
- **ctrlrange narrower than joint range**: The NN can't command the full range of motion. For example, a knee with range `[0, 2.0]` rad but ctrlrange `[-1, 1]` can only target up to 1.0 rad — half the knee's bend.
- **ctrlrange centered but joint range asymmetric**: A hip with range `[-1.0, 0.8]` but ctrlrange `[-1, 1]` wastes 0.2 rad on the positive side.

**Rule: set each actuator's ctrlrange to exactly match its joint's range.**

```xml
<!-- BAD: default ctrlrange, mismatched -->
<position name="knee_motor" joint="knee" kp="90" />  <!-- ctrlrange="-1 1" but joint range="0 2.0" -->

<!-- GOOD: ctrlrange matches joint range -->
<position name="knee_motor" joint="knee" kp="90" ctrlrange="0 2.0" />
```

Verify after any joint range change:
```bash
# Print all joint ranges and actuator ctrlranges
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml')
for i in range(m.nu):
    jnt = m.actuator_trnid[i, 0]
    name = m.joint(jnt).name
    jlo, jhi = m.jnt_range[jnt]
    clo, chi = m.actuator_ctrlrange[i]
    match = 'OK' if abs(clo - jlo) < 0.01 and abs(chi - jhi) < 0.01 else 'MISMATCH'
    print(f'{name:25s} joint=[{jlo:+.2f}, {jhi:+.2f}]  ctrl=[{clo:+.2f}, {chi:+.2f}]  {match}')
"
```

## 11. Gait Phase Period — Scale to Leg Length

The gait phase input encodes where the robot is in its stride cycle. The period must match the robot's natural stride dynamics, which scale with leg length.

**Pendulum model**: the natural swing period of a leg is `T = 2π * sqrt(L/g)`, where L is leg length (hip to ground).

| Robot | Leg length | Natural period | Good starting period |
|-------|-----------|---------------|---------------------|
| Duck biped | 0.17m | 0.83s | 0.6s |
| Child biped | 0.46m | 1.36s | 0.85s |

Real human walking is faster than the natural pendulum period (we push off), typically 60-70% of T.

**Don't copy gait period from a different-sized robot.** A period that's too fast forces the policy to fight the physics; too slow wastes the phase signal.

Rule of thumb: `gait_phase_period ≈ 0.65 * 2π * sqrt(leg_length / 9.81)`

## 12. Quick Validation Sequence

After building or modifying a bot:

```bash
# 1. Does it load?
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml'); print('OK')"

# 2. Does it stand?
uv run python stability_test.py --scene bots/BOTNAME/scene.xml --verbose

# 3. Does it look right? (check for interpenetration, floating, comically fast joints)
uv run mjpython main.py view --bot BOTNAME

# 4. Are joint speeds realistic?
# Compute terminal_vel = kp * max_error / damping for each joint.
# Multiply by lever arm to get tip speed. Should be < 3 m/s for walking bots.

# 5. Does training pipeline work?
uv run mjpython main.py train --bot BOTNAME --smoketest
```
