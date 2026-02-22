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

## 4. Motor Gains (kp) — Scale to Body Size

Position actuator spring gain (kp) must match the robot's mass and lever arms:

- **Too high**: joints snap violently to targets, creating oscillation and flailing
- **Too low**: robot sags, can't hold standing pose, sluggish response

Rough guideline: for a ~18 kg robot with 8cm upper legs:
- Hip/knee: kp = 30-50
- Ankle: kp = 20-35
- Hip abduction: kp = 20-30

The original biped (z=0.40, 12 kg torso, 12 cm legs) used kp=100. The duck biped (z=0.22, 10 kg torso, 8 cm legs) needs kp=40. **Smaller body = lower kp.**

## 5. Joint Damping — Prevent Oscillation

Damping resists joint velocity. Too low = underdamped oscillation. Too high = sluggish movement.

The damping ratio ζ = damping / (2 * sqrt(kp * I)), where I is the effective inertia. For RL, aim for ζ ≈ 0.3-0.5 (slightly underdamped, responsive but not wild).

Practical values for the duck biped (kp=40):
- Hip: damping = 3.0
- Knee: damping = 2.0
- Ankle: damping = 0.8
- Hip abduction: damping = 1.0

**Rule of thumb**: damping should be roughly 5-10% of kp for small robots.

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

## 10. Quick Validation Sequence

After building or modifying a bot:

```bash
# 1. Does it load?
uv run python -c "import mujoco; mujoco.MjModel.from_xml_path('bots/BOTNAME/scene.xml'); print('OK')"

# 2. Does it stand?
uv run python stability_test.py --scene bots/BOTNAME/scene.xml --verbose

# 3. Does it look right?
uv run mjpython main.py view --bot BOTNAME

# 4. Does training pipeline work?
uv run mjpython main.py train --bot BOTNAME --smoketest
```
