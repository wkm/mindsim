# Bot Validation: simplebiped
Generated: 2026-02-21

## Model Summary

| Property | Value |
|----------|-------|
| Bodies | 13 |
| Joints | 9 |
| Actuators | 8 |
| Sensors | 18 |
| Total mass | 13.01 kg |
| Timestep | 0.002 s |

## Rest State (all ctrl = 0)

### Body Positions

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.000 | -0.025 | 0.223 |
| left_upper_leg | -0.140 | -0.019 | 0.174 |
| left_lower_leg | -0.140 | -0.010 | 0.094 |
| left_foot | -0.140 | -0.003 | 0.035 |
| right_upper_leg | 0.140 | -0.019 | 0.174 |
| right_lower_leg | 0.140 | -0.010 | 0.094 |
| right_foot | 0.140 | -0.003 | 0.035 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

### Site Positions

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.000 | -0.025 | 0.223 |

### Contacts

- floor <-> left_lower_leg (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> right_lower_leg (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)

## Actuator Sweep

### left_hip_abd_motor
- **Joint**: left_hip_abd (hinge, axis=[0, 1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 25

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.413 | 0.587 |
| MAX | +1.000 | +0.413 | 0.587 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.117 | -0.006 | 0.225 |
| left_upper_leg | -0.024 | -0.003 | 0.175 |
| left_lower_leg | -0.056 | -0.001 | 0.102 |
| left_foot | -0.080 | 0.000 | 0.047 |
| right_upper_leg | 0.256 | -0.007 | 0.174 |
| right_lower_leg | 0.256 | -0.003 | 0.094 |
| right_foot | 0.255 | -0.001 | 0.035 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.117 | -0.006 | 0.225 |

**Contacts at MAX:**
- floor <-> left_foot (depth=-0.0008)
- floor <-> left_foot (depth=-0.0008)
- floor <-> right_lower_leg (depth=-0.0005)

**Contacts at MIN:**
- floor <-> left_foot (depth=-0.0008)
- floor <-> left_foot (depth=-0.0010)
- floor <-> right_lower_leg (depth=-0.0004)

### left_hip_motor
- **Joint**: left_hip (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 40

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.971 | 0.029 |
| MAX | +1.000 | +0.519 | 0.481 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | -0.046 | 0.173 | 0.139 |
| left_upper_leg | -0.189 | 0.137 | 0.159 |
| left_lower_leg | -0.207 | 0.080 | 0.106 |
| left_foot | -0.220 | 0.037 | 0.066 |
| right_upper_leg | 0.082 | 0.113 | 0.095 |
| right_lower_leg | 0.071 | 0.035 | 0.078 |
| right_foot | 0.063 | -0.023 | 0.065 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | -0.046 | 0.173 | 0.139 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0007)
- floor <-> right_foot (depth=-0.0005)

**Contacts at MIN:**
- floor <-> torso (depth=-0.0001)
- floor <-> left_upper_leg (depth=-0.0001)
- floor <-> right_foot (depth=-0.0009)

### left_knee_motor
- **Joint**: left_knee (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 40

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.065 | 0.935 |
| MAX | +1.000 | +1.033 | 0.033 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | -0.030 | 0.194 | 0.134 |
| left_upper_leg | -0.060 | 0.049 | 0.125 |
| left_lower_leg | 0.005 | 0.011 | 0.097 |
| left_foot | 0.012 | 0.003 | 0.038 |
| right_upper_leg | 0.080 | 0.290 | 0.106 |
| right_lower_leg | 0.145 | 0.251 | 0.080 |
| right_foot | 0.193 | 0.222 | 0.060 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | -0.030 | 0.194 | 0.134 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0001)
- floor <-> left_foot (depth=-0.0009)
- floor <-> right_foot (depth=-0.0006)

**Contacts at MIN:**
- floor <-> left_lower_leg (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> right_lower_leg (depth=-0.0003)
- floor <-> right_foot (depth=-0.0006)
- floor <-> right_foot (depth=-0.0006)

### left_ankle_motor
- **Joint**: left_ankle (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 30

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.433 | 0.567 |
| MAX | +1.000 | +0.433 | 0.567 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | -0.007 | 0.214 | 0.131 |
| left_upper_leg | -0.142 | 0.155 | 0.110 |
| left_lower_leg | -0.135 | 0.082 | 0.079 |
| left_foot | -0.130 | 0.027 | 0.055 |
| right_upper_leg | 0.136 | 0.183 | 0.110 |
| right_lower_leg | 0.143 | 0.109 | 0.080 |
| right_foot | 0.149 | 0.054 | 0.057 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | -0.007 | 0.214 | 0.131 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)

**Contacts at MIN:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0005)
- floor <-> right_foot (depth=-0.0004)

### right_hip_abd_motor
- **Joint**: right_hip_abd (hinge, axis=[0, 1, 0])
- **Range**: [-1.000, 1.000] | **kp**: 25

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.413 | 0.587 |
| MAX | +1.000 | +0.413 | 0.587 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | -0.001 | 0.000 | 0.225 |
| left_upper_leg | -0.141 | 0.000 | 0.175 |
| left_lower_leg | -0.140 | 0.001 | 0.095 |
| left_foot | -0.140 | 0.001 | 0.035 |
| right_upper_leg | 0.139 | 0.000 | 0.175 |
| right_lower_leg | 0.107 | 0.000 | 0.102 |
| right_foot | 0.083 | 0.000 | 0.047 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | -0.001 | 0.000 | 0.225 |

**Contacts at MAX:**
- floor <-> left_lower_leg (depth=-0.0004)
- floor <-> right_foot (depth=-0.0008)
- floor <-> right_foot (depth=-0.0010)

**Contacts at MIN:**
- floor <-> left_lower_leg (depth=-0.0005)
- floor <-> right_foot (depth=-0.0008)
- floor <-> right_foot (depth=-0.0008)

### right_hip_motor
- **Joint**: right_hip (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 40

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.971 | 0.029 |
| MAX | +1.000 | +0.519 | 0.481 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.046 | 0.173 | 0.139 |
| left_upper_leg | -0.082 | 0.113 | 0.095 |
| left_lower_leg | -0.071 | 0.035 | 0.078 |
| left_foot | -0.063 | -0.023 | 0.065 |
| right_upper_leg | 0.189 | 0.137 | 0.159 |
| right_lower_leg | 0.207 | 0.080 | 0.106 |
| right_foot | 0.220 | 0.037 | 0.066 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.046 | 0.173 | 0.139 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0005)
- floor <-> right_foot (depth=-0.0007)

**Contacts at MIN:**
- floor <-> torso (depth=-0.0001)
- floor <-> left_foot (depth=-0.0009)
- floor <-> right_upper_leg (depth=-0.0001)

### right_knee_motor
- **Joint**: right_knee (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 40

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.065 | 0.935 |
| MAX | +1.000 | +1.033 | 0.033 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.030 | 0.194 | 0.134 |
| left_upper_leg | -0.080 | 0.290 | 0.106 |
| left_lower_leg | -0.145 | 0.251 | 0.080 |
| left_foot | -0.193 | 0.222 | 0.060 |
| right_upper_leg | 0.060 | 0.049 | 0.125 |
| right_lower_leg | -0.005 | 0.011 | 0.097 |
| right_foot | -0.012 | 0.003 | 0.038 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.030 | 0.194 | 0.134 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0001)
- floor <-> left_foot (depth=-0.0006)
- floor <-> right_foot (depth=-0.0009)

**Contacts at MIN:**
- floor <-> left_lower_leg (depth=-0.0003)
- floor <-> left_foot (depth=-0.0006)
- floor <-> left_foot (depth=-0.0006)
- floor <-> right_lower_leg (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)

### right_ankle_motor
- **Joint**: right_ankle (hinge, axis=[1, 0, 0])
- **Range**: [-1.000, 1.000] | **kp**: 30

| | Target | Actual | Error |
|-----|--------|--------|-------|
| MIN | -1.000 | -0.433 | 0.567 |
| MAX | +1.000 | +0.433 | 0.567 |

**Body positions at MAX:**

| Body | X | Y | Z |
|------|------|------|------|
| base | 0.007 | 0.214 | 0.131 |
| left_upper_leg | -0.136 | 0.183 | 0.110 |
| left_lower_leg | -0.143 | 0.109 | 0.080 |
| left_foot | -0.149 | 0.054 | 0.057 |
| right_upper_leg | 0.142 | 0.155 | 0.110 |
| right_lower_leg | 0.135 | 0.082 | 0.079 |
| right_foot | 0.130 | 0.027 | 0.055 |
| target | 0.000 | 2.000 | 0.080 |
| distractor_0 | 0.000 | 0.000 | -1.000 |
| distractor_1 | 0.000 | 0.000 | -1.000 |
| distractor_2 | 0.000 | 0.000 | -1.000 |
| distractor_3 | 0.000 | 0.000 | -1.000 |

**Site positions at MAX:**

| Site | X | Y | Z |
|------|------|------|------|
| imu | 0.007 | 0.214 | 0.131 |

**Contacts at MAX:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)

**Contacts at MIN:**
- floor <-> torso (depth=-0.0002)
- floor <-> left_foot (depth=-0.0004)
- floor <-> left_foot (depth=-0.0005)
- floor <-> right_foot (depth=-0.0004)
- floor <-> right_foot (depth=-0.0004)

