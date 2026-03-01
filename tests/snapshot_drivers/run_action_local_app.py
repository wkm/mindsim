"""Driver app for RunActionScreen snapshot test (local run)."""

from pathlib import Path

from textual.app import App

from run_manager import RunInfo
from tui.screens.run_action import RunActionScreen

MOCK_RUN_INFO = RunInfo(
    name="s2w-lstm-0301-1045",
    bot_name="simple2wheeler",
    policy_type="lstm",
    algorithm="ppo",
    scene_path="bots/simple2wheeler/scene.xml",
    status="completed",
    batch_idx=5000,
    episode_count=12000,
    curriculum_stage=3,
    created_at="2026-03-01T10:45:00",
    wandb_url="https://wandb.ai/mindsim/s2w-lstm-0301-1045",
    wandb_id="abc123",
)


class RunActionLocalApp(App):
    def on_mount(self) -> None:
        self.push_screen(
            RunActionScreen(
                run_dir=Path("runs/s2w-lstm-0301-1045"),
                run_info=MOCK_RUN_INFO,
                cloud_only=False,
            )
        )


if __name__ == "__main__":
    RunActionLocalApp().run()
