"""
Stability measurement framework for biped models.

Runs three tests to quantify a biped's inherent stability:
1. Passive Standing — how long can it stand with zero control?
2. Perturbation Resistance — how much force/joint movement before falling?
3. Mobility-Stability Tradeoff — how much leg movement is possible while staying upright?

Usage:
    uv run python stability_test.py [--scene PATH] [--verbose] [--video]
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# --- Video recording ---

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30


class VideoRecorder:
    """Records MuJoCo simulation frames to MP4 via mediapy."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_path: str | Path,
        fps: int = VIDEO_FPS,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
    ):
        self.data = data
        self.output_path = str(output_path)
        self.fps = fps
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.frames: list[np.ndarray] = []

        # Configure a side-view free camera
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.0, 0.0, 0.4]  # hip height
        self.camera.distance = 2.5
        self.camera.azimuth = 135  # 3/4 view (front-left)
        self.camera.elevation = -20

    def capture(self):
        """Capture one frame from the current simulation state."""
        self.renderer.update_scene(self.data, camera=self.camera)
        self.frames.append(self.renderer.render().copy())

    def save(self):
        """Write collected frames to MP4."""
        if not self.frames:
            return
        import mediapy

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        mediapy.write_video(self.output_path, self.frames, fps=self.fps)
        print(f"  Video: {self.output_path} ({len(self.frames)} frames, {len(self.frames)/self.fps:.1f}s)")

    def close(self):
        self.renderer.close()


def _frame_interval(dt: float, fps: int) -> int:
    """Physics steps per video frame."""
    return max(1, int(1.0 / (fps * dt)))


# --- Fall detection helpers ---


def get_torso_up_z(data: mujoco.MjData, body_id: int) -> float:
    """Get Z component of torso's up vector (1.0 = perfectly upright)."""
    w, x, y, z = data.xquat[body_id]
    return 1 - 2 * (x * x + y * y)


def get_torso_height(data: mujoco.MjData, body_id: int) -> float:
    """Get torso Z position."""
    return data.xpos[body_id][2]


def has_fallen(
    data: mujoco.MjData, body_id: int, min_height: float, min_up_z: float
) -> bool:
    """Check if the biped has fallen over."""
    return (
        get_torso_height(data, body_id) < min_height
        or get_torso_up_z(data, body_id) < min_up_z
    )


def compute_support_polygon_margin(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    floor_geom_id: int,
    foot_geom_ids: set[int],
) -> float:
    """
    Compute static stability margin: minimum distance from CoM projection
    to the edge of the support polygon (convex hull of foot contacts).

    Returns 0 if no foot contacts, negative if CoM is outside support polygon.
    """
    foot_contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        if g1 == floor_geom_id and g2 in foot_geom_ids:
            foot_contacts.append(contact.pos[:2].copy())
        elif g2 == floor_geom_id and g1 in foot_geom_ids:
            foot_contacts.append(contact.pos[:2].copy())

    if len(foot_contacts) < 2:
        return 0.0

    foot_contacts = np.array(foot_contacts)
    com_xy = data.subtree_com[0][:2]  # World CoM projected onto ground

    # Use convex hull of ALL foot contact points (both feet together)
    # For a simple margin: distance from CoM to nearest edge of bounding box
    min_xy = foot_contacts.min(axis=0)
    max_xy = foot_contacts.max(axis=0)

    # Ensure bounding box has non-zero area (avoid degenerate cases)
    if (max_xy - min_xy).min() < 1e-6:
        return 0.0

    # Distance from CoM to nearest bounding box edge (negative if outside)
    dx = min(com_xy[0] - min_xy[0], max_xy[0] - com_xy[0])
    dy = min(com_xy[1] - min_xy[1], max_xy[1] - com_xy[1])

    # If both positive, CoM is inside the polygon
    if dx >= 0 and dy >= 0:
        return min(dx, dy)
    # If either negative, CoM is outside
    return min(dx, dy)


# --- Model loading ---


def load_model(scene_path: str) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Load model and discover key IDs."""
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    foot_geom_ids = set()
    for name in ("left_foot", "right_foot"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            foot_geom_ids.add(gid)

    ids = {
        "body": body_id,
        "floor": floor_geom_id,
        "feet": foot_geom_ids,
    }
    return model, data, ids


def reset_to_standing(model: mujoco.MjModel, data: mujoco.MjData):
    """Reset to default pose (standing) and run forward kinematics."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


# --- Test 1: Passive Standing ---


@dataclass
class PassiveStandingResult:
    time_to_fall: float  # seconds until fall (-1 = survived full duration)
    survived: bool
    min_up_z: float  # minimum uprightness during test
    min_height: float  # minimum torso height during test
    avg_stability_margin: float  # average static stability margin
    initial_height: float
    initial_up_z: float


def test_passive_standing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    duration: float = 10.0,
    settle_time: float = 1.0,
    fall_height_frac: float = 0.5,
    fall_up_z: float = 0.3,
    verbose: bool = False,
    video: VideoRecorder | None = None,
) -> PassiveStandingResult:
    """
    Test 1: Can the biped stand with zero motor input?

    Simulates for `duration` seconds with all controls at zero.
    Includes a settle_time period where the robot is allowed to drop
    onto the floor and stabilize before measurement begins.
    Tracks time-to-fall, minimum uprightness, and stability margin.
    """
    reset_to_standing(model, data)
    dt = model.opt.timestep
    frame_every = _frame_interval(dt, VIDEO_FPS)

    # Let the robot settle (drop onto floor, absorb impact)
    settle_steps = int(settle_time / dt)
    for s in range(settle_steps):
        mujoco.mj_step(model, data)
        if video and s % frame_every == 0:
            video.capture()

    # Measure from settled state
    initial_height = get_torso_height(data, ids["body"])
    initial_up_z = get_torso_up_z(data, ids["body"])
    fall_height = initial_height * fall_height_frac

    if verbose:
        print(
            f"  After {settle_time}s settle: height={initial_height:.4f}m, up_z={initial_up_z:.4f}"
        )

    # Already fallen during settling?
    if has_fallen(data, ids["body"], fall_height, fall_up_z):
        if video:
            video.save()
        return PassiveStandingResult(
            time_to_fall=0.0,
            survived=False,
            min_up_z=initial_up_z,
            min_height=initial_height,
            avg_stability_margin=0.0,
            initial_height=initial_height,
            initial_up_z=initial_up_z,
        )

    min_up_z = initial_up_z
    min_height = initial_height
    margins = []
    time_to_fall = -1.0
    t = 0.0
    step = 0

    while t < duration:
        mujoco.mj_step(model, data)
        t += dt
        step += 1

        if video and step % frame_every == 0:
            video.capture()

        up_z = get_torso_up_z(data, ids["body"])
        height = get_torso_height(data, ids["body"])
        min_up_z = min(min_up_z, up_z)
        min_height = min(min_height, height)

        if step % 50 == 0:
            margin = compute_support_polygon_margin(
                model, data, ids["body"], ids["floor"], ids["feet"]
            )
            margins.append(margin)

        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            time_to_fall = t
            # Capture a few more frames of the fall
            if video:
                for _ in range(VIDEO_FPS):  # ~1s of post-fall
                    mujoco.mj_step(model, data)
                    video.capture()
            break

    if video:
        video.save()

    if verbose:
        status = (
            f"FELL at {time_to_fall:.3f}s"
            if time_to_fall > 0
            else f"SURVIVED {duration}s"
        )
        print(f"  Passive standing: {status}")
        print(
            f"    Initial height: {initial_height:.4f}m, min height: {min_height:.4f}m"
        )
        print(f"    Initial up_z: {initial_up_z:.4f}, min up_z: {min_up_z:.4f}")
        if margins:
            print(f"    Avg stability margin: {np.mean(margins):.4f}m")

    return PassiveStandingResult(
        time_to_fall=time_to_fall,
        survived=time_to_fall < 0,
        min_up_z=min_up_z,
        min_height=min_height,
        avg_stability_margin=float(np.mean(margins)) if margins else 0.0,
        initial_height=initial_height,
        initial_up_z=initial_up_z,
    )


# --- Test 2: Perturbation Resistance ---


@dataclass
class PerturbationResult:
    max_lateral_impulse: float  # max recoverable lateral force impulse (N*s)
    max_forward_impulse: float  # max recoverable forward force impulse (N*s)
    max_joint_perturbation: float  # max sinusoidal amplitude before fall (rad)
    lateral_impulses_tested: list[float]
    lateral_survived: list[bool]
    forward_impulses_tested: list[float]
    forward_survived: list[bool]


def _test_impulse(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    force_direction: np.ndarray,
    impulse_magnitude: float,
    recovery_time: float = 3.0,
    fall_height_frac: float = 0.5,
    fall_up_z: float = 0.3,
    video: VideoRecorder | None = None,
) -> bool:
    """Apply an impulse to the torso and check if the biped recovers."""
    reset_to_standing(model, data)
    dt = model.opt.timestep
    frame_every = _frame_interval(dt, VIDEO_FPS)

    initial_height = get_torso_height(data, ids["body"])
    fall_height = initial_height * fall_height_frac

    # Let it settle first (0.5s)
    settle_steps = int(0.5 / dt)
    for s in range(settle_steps):
        mujoco.mj_step(model, data)
        if video and s % frame_every == 0:
            video.capture()
        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            if video:
                video.save()
            return False  # Can't even stand

    # Apply impulse as a brief force over a few timesteps
    impulse_duration = 0.02  # 20ms burst
    impulse_steps = max(1, int(impulse_duration / dt))
    force = force_direction * (impulse_magnitude / impulse_duration)

    for s in range(impulse_steps):
        data.xfrc_applied[ids["body"], :3] = force
        mujoco.mj_step(model, data)
        if video:
            video.capture()
    data.xfrc_applied[ids["body"], :3] = 0  # Remove force

    # Check if it recovers over recovery_time
    recovery_steps = int(recovery_time / dt)
    step = 0
    for s in range(recovery_steps):
        mujoco.mj_step(model, data)
        step += 1
        if video and step % frame_every == 0:
            video.capture()
        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            # Capture post-fall
            if video:
                for _ in range(VIDEO_FPS):
                    mujoco.mj_step(model, data)
                    video.capture()
                video.save()
            return False

    if video:
        video.save()
    return True


def _test_joint_perturbation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    amplitude: float,
    frequency: float = 2.0,
    duration: float = 5.0,
    fall_height_frac: float = 0.5,
    fall_up_z: float = 0.3,
) -> bool:
    """Apply sinusoidal perturbations to all joints and check survival."""
    reset_to_standing(model, data)
    dt = model.opt.timestep

    initial_height = get_torso_height(data, ids["body"])
    fall_height = initial_height * fall_height_frac

    # Let it settle first
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(model, data)
        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            return False

    t = 0.0
    while t < duration:
        # Apply sinusoidal control to all actuators (alternating legs)
        for i in range(model.nu):
            phase = np.pi * (i % 2)  # Alternate legs
            data.ctrl[i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)

        mujoco.mj_step(model, data)
        t += dt

        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            return False

    return True


def test_perturbation_resistance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    verbose: bool = False,
    video_dir: str | None = None,
) -> PerturbationResult:
    """
    Test 2: How much perturbation before falling?

    Tests lateral and forward impulses at increasing magnitudes,
    and sinusoidal joint perturbations at increasing amplitudes.
    When video_dir is set, records the boundary cases (last-pass, first-fail).
    """
    # Lateral impulses (in X direction)
    lateral_magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    lateral_dir = np.array([1.0, 0.0, 0.0])
    lateral_survived = []
    max_lateral = 0.0

    for mag in lateral_magnitudes:
        survived = _test_impulse(model, data, ids, lateral_dir, mag)
        lateral_survived.append(survived)
        if survived:
            max_lateral = mag
        if verbose:
            sym = "OK" if survived else "FELL"
            print(f"  Lateral impulse {mag:5.1f} N*s: {sym}")
        if not survived:
            break

    # Record boundary videos for lateral
    if video_dir:
        if max_lateral > 0:
            rec = VideoRecorder(model, data, f"{video_dir}/02_perturbation_lateral_{max_lateral:.1f}Ns_pass.mp4")
            _test_impulse(model, data, ids, lateral_dir, max_lateral, video=rec)
            rec.close()
        fail_mag = lateral_magnitudes[len(lateral_survived) - 1] if not lateral_survived[-1] else None
        if fail_mag is not None:
            rec = VideoRecorder(model, data, f"{video_dir}/02_perturbation_lateral_{fail_mag:.1f}Ns_fail.mp4")
            _test_impulse(model, data, ids, lateral_dir, fail_mag, video=rec)
            rec.close()

    # Forward impulses (in -Y direction, biped faces -Y)
    forward_magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]
    forward_dir = np.array([0.0, -1.0, 0.0])
    forward_survived = []
    max_forward = 0.0

    for mag in forward_magnitudes:
        survived = _test_impulse(model, data, ids, forward_dir, mag)
        forward_survived.append(survived)
        if survived:
            max_forward = mag
        if verbose:
            sym = "OK" if survived else "FELL"
            print(f"  Forward impulse {mag:5.1f} N*s: {sym}")
        if not survived:
            break

    # Record boundary videos for forward
    if video_dir:
        if max_forward > 0:
            rec = VideoRecorder(model, data, f"{video_dir}/02_perturbation_forward_{max_forward:.1f}Ns_pass.mp4")
            _test_impulse(model, data, ids, forward_dir, max_forward, video=rec)
            rec.close()
        fail_mag = forward_magnitudes[len(forward_survived) - 1] if not forward_survived[-1] else None
        if fail_mag is not None:
            rec = VideoRecorder(model, data, f"{video_dir}/02_perturbation_forward_{fail_mag:.1f}Ns_fail.mp4")
            _test_impulse(model, data, ids, forward_dir, fail_mag, video=rec)
            rec.close()

    # Joint perturbation (sinusoidal) — no video, these are long
    joint_amplitudes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    max_joint = 0.0

    for amp in joint_amplitudes:
        survived = _test_joint_perturbation(model, data, ids, amp)
        if survived:
            max_joint = amp
        if verbose:
            sym = "OK" if survived else "FELL"
            print(f"  Joint perturbation {amp:.2f} rad: {sym}")
        if not survived:
            break

    return PerturbationResult(
        max_lateral_impulse=max_lateral,
        max_forward_impulse=max_forward,
        max_joint_perturbation=max_joint,
        lateral_impulses_tested=lateral_magnitudes[: len(lateral_survived)],
        lateral_survived=lateral_survived,
        forward_impulses_tested=forward_magnitudes[: len(forward_survived)],
        forward_survived=forward_survived,
    )


# --- Test 3: Mobility-Stability Tradeoff ---


@dataclass
class MobilityResult:
    """Results from the mobility-stability tradeoff test."""

    amplitudes: list[float]
    survival_times: list[float]  # seconds survived at each amplitude
    forward_distances: list[float]  # forward displacement achieved
    max_stable_amplitude: float  # largest amplitude that survived full duration
    best_forward_distance: float  # max forward distance while staying upright


def _run_gait_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    amp: float,
    duration: float,
    gait_frequency: float,
    fall_height_frac: float,
    fall_up_z: float,
    video: VideoRecorder | None = None,
) -> tuple[float, float, bool]:
    """Run a single gait trial. Returns (survival_time, forward_distance, fell)."""
    reset_to_standing(model, data)
    dt = model.opt.timestep
    frame_every = _frame_interval(dt, VIDEO_FPS)

    initial_height = get_torso_height(data, ids["body"])
    fall_height = initial_height * fall_height_frac

    # Settle
    settle_steps = int(0.3 / dt)
    for s in range(settle_steps):
        mujoco.mj_step(model, data)
        if video and s % frame_every == 0:
            video.capture()

    start_pos = data.xpos[ids["body"]].copy()
    t = 0.0
    fell = False
    step = 0

    while t < duration:
        phase = 2 * np.pi * gait_frequency * t
        for i in range(model.nu):
            motor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if motor_name is None:
                continue

            is_left = "left" in motor_name
            leg_phase = phase if is_left else phase + np.pi

            if "hip_abd" in motor_name:
                data.ctrl[i] = amp * 0.2 * np.sin(leg_phase)
            elif "hip" in motor_name:
                data.ctrl[i] = amp * np.sin(leg_phase)
            elif "knee" in motor_name:
                data.ctrl[i] = amp * 0.5 * max(0, np.sin(leg_phase - 0.5))
            elif "ankle" in motor_name:
                data.ctrl[i] = amp * 0.3 * np.sin(leg_phase + 0.3)

        mujoco.mj_step(model, data)
        t += dt
        step += 1

        if video and step % frame_every == 0:
            video.capture()

        if has_fallen(data, ids["body"], fall_height, fall_up_z):
            fell = True
            if video:
                for _ in range(VIDEO_FPS):
                    mujoco.mj_step(model, data)
                    video.capture()
            break

    if video:
        video.save()

    end_pos = data.xpos[ids["body"]].copy()
    forward_dist = -(end_pos[1] - start_pos[1])  # -Y is forward
    return t, forward_dist, fell


def test_mobility_tradeoff(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    duration: float = 5.0,
    gait_frequency: float = 1.5,
    fall_height_frac: float = 0.5,
    fall_up_z: float = 0.3,
    verbose: bool = False,
    video_dir: str | None = None,
) -> MobilityResult:
    """
    Test 3: Mobility-stability tradeoff.

    Applies a simple alternating gait pattern (sinusoidal hip + knee control)
    at increasing amplitudes. Measures how long the robot survives and how
    far forward it travels.
    """
    amplitudes = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    survival_times: list[float] = []
    forward_distances: list[float] = []
    max_stable_amp = 0.0
    best_forward = 0.0

    # Record which amplitudes to video: lowest, boundary, and a high-amp fall
    video_amps: set[float] = set()
    if video_dir:
        video_amps.add(amplitudes[0])  # baseline

    for amp in amplitudes:
        t, forward_dist, fell = _run_gait_trial(
            model, data, ids, amp, duration, gait_frequency,
            fall_height_frac, fall_up_z,
        )

        survival_times.append(t)
        forward_distances.append(forward_dist)

        if not fell:
            max_stable_amp = amp
        if forward_dist > best_forward and (not fell or t > 2.0):
            best_forward = forward_dist

        if verbose:
            status = f"survived {duration:.1f}s" if not fell else f"fell at {t:.2f}s"
            print(
                f"  Gait amplitude {amp:.2f}: {status}, forward: {forward_dist:+.3f}m"
            )

        # First failure: record this and the previous (last pass)
        if fell and video_dir and amp not in video_amps:
            video_amps.add(amp)
            # Previous amplitude was last pass (if it exists and survived)
            idx = amplitudes.index(amp)
            if idx > 0 and not survival_times[idx - 1] < duration:
                pass  # already fell at previous too
            elif idx > 0:
                video_amps.add(amplitudes[idx - 1])
            break  # stop after first fail for video selection

    # Continue remaining amplitudes (without video consideration)
    remaining_start = len(survival_times)
    for amp in amplitudes[remaining_start:]:
        t, forward_dist, fell = _run_gait_trial(
            model, data, ids, amp, duration, gait_frequency,
            fall_height_frac, fall_up_z,
        )
        survival_times.append(t)
        forward_distances.append(forward_dist)
        if not fell:
            max_stable_amp = amp
        if forward_dist > best_forward and (not fell or t > 2.0):
            best_forward = forward_dist
        if verbose:
            status = f"survived {duration:.1f}s" if not fell else f"fell at {t:.2f}s"
            print(f"  Gait amplitude {amp:.2f}: {status}, forward: {forward_dist:+.3f}m")

    # Now record the selected video amplitudes
    if video_dir and video_amps:
        for amp in sorted(video_amps):
            rec = VideoRecorder(model, data, f"{video_dir}/03_mobility_amp_{amp:.2f}.mp4")
            _run_gait_trial(
                model, data, ids, amp, duration, gait_frequency,
                fall_height_frac, fall_up_z, video=rec,
            )
            rec.close()

    return MobilityResult(
        amplitudes=amplitudes,
        survival_times=survival_times,
        forward_distances=forward_distances,
        max_stable_amplitude=max_stable_amp,
        best_forward_distance=best_forward,
    )


# --- Summary ---


@dataclass
class StabilityReport:
    scene_path: str
    passive: PassiveStandingResult
    perturbation: PerturbationResult
    mobility: MobilityResult

    def format_summary(self) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"STABILITY REPORT: {self.scene_path}")
        lines.append("=" * 60)

        # Passive standing
        p = self.passive
        if p.survived:
            lines.append("\n  1. Passive Standing:    PASS (survived 10s)")
        else:
            lines.append(f"\n  1. Passive Standing:    FAIL (fell at {p.time_to_fall:.3f}s)")
        lines.append(f"     Initial height:      {p.initial_height:.4f}m")
        lines.append(f"     Min height:          {p.min_height:.4f}m")
        lines.append(f"     Min uprightness:     {p.min_up_z:.4f}")
        lines.append(f"     Stability margin:    {p.avg_stability_margin:.4f}m")

        # Perturbation resistance
        r = self.perturbation
        lines.append("\n  2. Perturbation Resistance:")
        lines.append(f"     Max lateral impulse: {r.max_lateral_impulse:.1f} N*s")
        lines.append(f"     Max forward impulse: {r.max_forward_impulse:.1f} N*s")
        lines.append(f"     Max joint perturb:   {r.max_joint_perturbation:.2f} rad")

        # Mobility
        m = self.mobility
        lines.append("\n  3. Mobility-Stability Tradeoff:")
        lines.append(f"     Max stable gait amp: {m.max_stable_amplitude:.2f}")
        lines.append(f"     Best forward dist:   {m.best_forward_distance:.3f}m")

        # Overall score (composite metric for quick comparison)
        standing_score = 1.0 if p.survived else p.time_to_fall / 10.0
        impulse_score = min(1.0, r.max_lateral_impulse / 10.0)
        mobility_score = min(1.0, m.max_stable_amplitude / 0.5)
        overall = standing_score * 0.4 + impulse_score * 0.3 + mobility_score * 0.3

        lines.append(f"\n  OVERALL SCORE: {overall:.2f} / 1.00")
        lines.append(
            f"    (standing={standing_score:.2f} * 0.4 + "
            f"impulse={impulse_score:.2f} * 0.3 + "
            f"mobility={mobility_score:.2f} * 0.3)"
        )
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self):
        print("\n" + self.format_summary())


def run_stability_tests(
    scene_path: str, verbose: bool = False, video: bool = False
) -> StabilityReport:
    """Run all stability tests on the given scene."""
    print(f"\nLoading model: {scene_path}")
    model, data, ids = load_model(scene_path)

    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"  Timestep: {model.opt.timestep}s")

    # Compute total mass
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    torso_mass = model.body_mass[ids["body"]]
    print(
        f"  Total mass: {total_mass:.2f} kg (torso: {torso_mass:.2f} kg, {torso_mass / total_mass * 100:.0f}%)"
    )

    # Set up video output directory
    video_dir: str | None = None
    if video:
        bot_name = Path(scene_path).parent.name
        video_dir = f"stability_videos/{bot_name}"
        os.makedirs(video_dir, exist_ok=True)
        print(f"  Video output: {video_dir}/")

    print("\n--- Test 1: Passive Standing ---")
    if video:
        rec = VideoRecorder(model, data, f"{video_dir}/01_passive_standing.mp4")
        passive = test_passive_standing(model, data, ids, verbose=verbose, video=rec)
        rec.close()
    else:
        passive = test_passive_standing(model, data, ids, verbose=verbose)

    print("\n--- Test 2: Perturbation Resistance ---")
    perturbation = test_perturbation_resistance(
        model, data, ids, verbose=verbose, video_dir=video_dir
    )

    print("\n--- Test 3: Mobility-Stability Tradeoff ---")
    mobility = test_mobility_tradeoff(
        model, data, ids, verbose=verbose, video_dir=video_dir
    )

    report = StabilityReport(
        scene_path=scene_path,
        passive=passive,
        perturbation=perturbation,
        mobility=mobility,
    )
    report.print_summary()

    # Write report to file alongside videos
    if video_dir:
        report_path = f"{video_dir}/report.txt"
        with open(report_path, "w") as f:
            f.write(report.format_summary())
        print(f"\n  Report saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Biped stability measurement framework"
    )
    parser.add_argument(
        "--scene", default="bots/simplebiped/scene.xml", help="Path to scene XML"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed per-test output"
    )
    parser.add_argument(
        "--video", action="store_true", help="Render MP4 videos of each test"
    )
    args = parser.parse_args()

    run_stability_tests(args.scene, verbose=args.verbose, video=args.video)


if __name__ == "__main__":
    main()
