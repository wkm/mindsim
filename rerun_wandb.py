"""
Integration between Rerun and Weights & Biases.

Links Rerun eval episode recordings to wandb training runs for easy navigation
from a point in training to the corresponding visualization.

Uses deterministic eval episodes (mean actions, no sampling) to show
true policy capability without exploration noise.
"""

import os

import rerun as rr

import rerun_logger
import wandb
from training_blueprint import create_training_blueprint


class RerunWandbLogger:
    """
    Manages Rerun recordings linked to wandb runs.

    Saves per-episode .rrd files with naming: <run_dir>/recordings/episode_<N>.rrd
    (or legacy recordings/<run_name>_<run_id>/ when no run_dir is given).
    Logs paths to wandb for cross-referencing.
    """

    def __init__(self, recordings_dir: str = "recordings", live: bool = True,
                 run_dir: str | None = None):
        """
        Initialize the logger.

        Args:
            recordings_dir: Base directory for storing .rrd files (legacy fallback)
            live: If True, spawn Rerun viewer and stream data live
            run_dir: Run directory path. When set, saves to run_dir/recordings/.
                     When None, falls back to recordings/<run_name>_<run_id>/.
        """
        if wandb.run is None:
            raise RuntimeError(
                "wandb.init() must be called before creating RerunWandbLogger"
            )

        self.recordings_dir = recordings_dir
        self.run_id = wandb.run.id
        self.run_name = wandb.run.name
        if run_dir is not None:
            self.run_dir = os.path.join(run_dir, "recordings")
        else:
            self.run_dir = os.path.join(recordings_dir, f"{self.run_name}_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.live = live

        self.current_episode = None
        self.rrd_path = None
        self.is_recording = False
        self._spawned = False

    def start_episode(self, episode: int, env, namespace: str = "eval", show_camera: bool = True):
        """
        Start a new Rerun recording for this episode.

        Args:
            episode: Episode number
            env: Training environment instance
            namespace: Rerun namespace for logging
            show_camera: If False, hide the camera view and skip camera logging

        Returns:
            Path to the .rrd file being recorded
        """
        self.current_episode = episode
        self.rrd_path = os.path.join(self.run_dir, f"episode_{episode:05d}.rrd")

        # Initialize Rerun: application_id = run name, recording = episode number
        rr.init(self.run_name, recording_id=f"{self.run_id}-ep{episode}")

        # Spawn viewer on first episode (without connecting, we'll use set_sinks)
        if self.live and not self._spawned:
            rr.spawn(connect=False)
            self._spawned = True

        # Set up sinks: always save to disk, optionally stream live
        sinks = [rr.FileSink(self.rrd_path)]
        if self.live:
            try:
                sinks.append(rr.GrpcSink())
            except Exception:
                pass  # Viewer may have been closed; file sink still works
        rr.set_sinks(*sinks)

        # Set recording name to just the episode number
        rr.send_recording_name(f"episode {episode}")

        # Embed blueprint into recording
        control_fps = round(1.0 / env.action_dt)
        rr.send_blueprint(create_training_blueprint(control_fps=control_fps, show_camera=show_camera))

        # Log wandb context into Rerun for reverse lookup
        rr.log(
            "meta/wandb",
            rr.TextDocument(
                f"Run: {self.run_name}\n"
                f"ID: {self.run_id}\n"
                f"Episode: {episode}\n"
                f"URL: {wandb.run.url}"
            ),
            static=True,
        )

        # Set up the 3D scene (pass arena boundary if available)
        arena_boundary = getattr(env, "arena_boundary", None)
        rerun_logger.setup_scene(
            env, namespace=namespace, arena_boundary=arena_boundary, show_camera=show_camera
        )

        self.is_recording = True
        return self.rrd_path

    def finish_episode(self, episode_data: dict = None, upload_artifact: bool = False):
        """
        Finish recording and log reference to wandb.

        Args:
            episode_data: Optional dict with episode stats to log
            upload_artifact: If True, upload .rrd as wandb artifact (for remote access)
        """
        if not self.is_recording:
            return

        # Log recording path to wandb
        log_data = {
            "rerun/recording_path": self.rrd_path,
            "rerun/episode": self.current_episode,
        }

        if episode_data:
            log_data["rerun/episode_reward"] = episode_data.get("total_reward")
            log_data["rerun/episode_distance"] = episode_data.get("final_distance")

        wandb.log(log_data)

        # Optionally upload as artifact for remote access
        if upload_artifact:
            artifact = wandb.Artifact(
                f"rerun-episode-{self.current_episode:05d}",
                type="rerun-recording",
                metadata={
                    "episode": self.current_episode,
                    "run_id": self.run_id,
                },
            )
            artifact.add_file(self.rrd_path)
            wandb.log_artifact(artifact)

        # Flush recording and release file handles / background threads.
        # Without this, each start_episode() accumulates leaked threads.
        rr.disconnect()

        self.is_recording = False

    def get_recording_path(self, episode: int) -> str:
        """Get the expected path for an episode's recording."""
        return os.path.join(self.run_dir, f"episode_{episode:05d}.rrd")

    def list_recordings(self) -> list[str]:
        """List all .rrd files for this run."""
        if not os.path.exists(self.run_dir):
            return []
        return sorted(
            [
                os.path.join(self.run_dir, f)
                for f in os.listdir(self.run_dir)
                if f.endswith(".rrd")
            ]
        )
