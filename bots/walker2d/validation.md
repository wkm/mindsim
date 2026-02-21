# Bot Validation: walker2d
Generated: 2026-02-21

## Model Summary

| Property | Value |
|----------|-------|
| Bodies | 13 |
| Joints | 9 |
| Actuators | 6 |
| Sensors | 14 |
| Total mass | 28.68 kg |
| Timestep | 0.002 s |

## Rest State (all ctrl = 0)

### Body Positions

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.296 | 0.000 | 0.238 |
| thigh | 0.364 | 0.000 | 0.426 |
| leg | 0.381 | 0.000 | 0.148 |
| foot | 0.226 | 0.000 | 0.060 |
| thigh_left | 0.364 | 0.000 | 0.426 |
| leg_left | 0.381 | 0.000 | 0.148 |
| foot_left | 0.226 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

### Site Positions

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.296 | 0.000 | 0.238 |

### Contacts

- floor <-> torso_geom (depth=-0.0002)
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0002)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> left_foot (depth=-0.0002)

## Actuator Sweep

### thigh_motor
- **Joint**: thigh_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -2.637 | 1.637 |
| MAX | +1.000 | +0.021 | 0.979 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.251 | 0.000 | 0.569 |
| thigh | 0.371 | 0.000 | 0.409 |
| leg | 0.417 | 0.000 | 0.149 |
| foot | 0.264 | 0.000 | 0.060 |
| thigh_left | 0.371 | 0.000 | 0.409 |
| leg_left | 0.412 | 0.000 | 0.148 |
| foot_left | 0.257 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.251 | 0.000 | 0.569 |

**Contacts at MAX:**
- floor <-> right_foot (depth=-0.0004)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> left_foot (depth=-0.0002)

**Contacts at MIN:**
- floor <-> right_foot (depth=-0.0002)
- floor <-> right_foot (depth=-0.0005)
- floor <-> left_foot (depth=-0.0002)
- floor <-> left_foot (depth=-0.0001)

### leg_motor
- **Joint**: leg_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -2.636 | 1.636 |
| MAX | +1.000 | +0.018 | 0.982 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.930 | 0.000 | 0.181 |
| thigh | 1.081 | 0.000 | 0.050 |
| leg | 0.393 | 0.000 | 0.180 |
| foot | 0.262 | 0.000 | 0.060 |
| thigh_left | 1.081 | 0.000 | 0.050 |
| leg_left | 0.401 | 0.000 | 0.148 |
| foot_left | 0.246 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.930 | 0.000 | 0.181 |

**Contacts at MAX:**
- floor <-> torso_geom (depth=-0.0001)
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0003)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> left_foot (depth=-0.0002)

**Contacts at MIN:**
- floor <-> thigh_geom (depth=-0.0003)
- floor <-> thigh_left_geom (depth=-0.0000)
- floor <-> left_foot (depth=-0.0002)

### foot_motor
- **Joint**: foot_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.823 | 0.177 |
| MAX | +1.000 | +0.824 | 0.176 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.408 | 0.000 | 0.249 |
| thigh | 0.420 | 0.000 | 0.449 |
| leg | 0.413 | 0.000 | 0.150 |
| foot | 0.269 | 0.000 | 0.060 |
| thigh_left | 0.420 | 0.000 | 0.449 |
| leg_left | 0.397 | 0.000 | 0.148 |
| foot_left | 0.242 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.408 | 0.000 | 0.249 |

**Contacts at MAX:**
- floor <-> torso_geom (depth=-0.0002)
- floor <-> right_foot (depth=-0.0002)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> left_foot (depth=-0.0002)

**Contacts at MIN:**
- floor <-> torso_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0003)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0000)

### thigh_left_motor
- **Joint**: thigh_left_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -2.637 | 1.637 |
| MAX | +1.000 | +0.021 | 0.979 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.251 | 0.000 | 0.569 |
| thigh | 0.371 | 0.000 | 0.409 |
| leg | 0.412 | 0.000 | 0.148 |
| foot | 0.257 | 0.000 | 0.060 |
| thigh_left | 0.371 | 0.000 | 0.409 |
| leg_left | 0.417 | 0.000 | 0.149 |
| foot_left | 0.264 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.251 | 0.000 | 0.569 |

**Contacts at MAX:**
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0002)
- floor <-> left_foot (depth=-0.0004)

**Contacts at MIN:**
- floor <-> right_foot (depth=-0.0002)
- floor <-> right_foot (depth=-0.0001)
- floor <-> left_foot (depth=-0.0002)
- floor <-> left_foot (depth=-0.0005)

### leg_left_motor
- **Joint**: leg_left_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -2.636 | 1.636 |
| MAX | +1.000 | +0.018 | 0.982 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.930 | 0.000 | 0.181 |
| thigh | 1.081 | 0.000 | 0.050 |
| leg | 0.401 | 0.000 | 0.148 |
| foot | 0.246 | 0.000 | 0.060 |
| thigh_left | 1.081 | 0.000 | 0.050 |
| leg_left | 0.393 | 0.000 | 0.180 |
| foot_left | 0.262 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.930 | 0.000 | 0.181 |

**Contacts at MAX:**
- floor <-> torso_geom (depth=-0.0001)
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0002)
- floor <-> thigh_left_geom (depth=-0.0001)
- floor <-> left_foot (depth=-0.0003)

**Contacts at MIN:**
- floor <-> thigh_geom (depth=-0.0000)
- floor <-> right_foot (depth=-0.0002)
- floor <-> thigh_left_geom (depth=-0.0003)

### foot_left_motor
- **Joint**: foot_left_joint (hinge, axis=[0, -1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 1

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.823 | 0.177 |
| MAX | +1.000 | +0.824 | 0.176 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.408 | 0.000 | 0.249 |
| thigh | 0.420 | 0.000 | 0.449 |
| leg | 0.397 | 0.000 | 0.148 |
| foot | 0.242 | 0.000 | 0.060 |
| thigh_left | 0.420 | 0.000 | 0.449 |
| leg_left | 0.413 | 0.000 | 0.150 |
| foot_left | 0.269 | 0.000 | 0.060 |
| target | 10.000 | 0.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.408 | 0.000 | 0.249 |

**Contacts at MAX:**
- floor <-> torso_geom (depth=-0.0002)
- floor <-> thigh_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0002)
- floor <-> left_foot (depth=-0.0002)

**Contacts at MIN:**
- floor <-> torso_geom (depth=-0.0001)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0000)
- floor <-> left_foot (depth=-0.0003)

