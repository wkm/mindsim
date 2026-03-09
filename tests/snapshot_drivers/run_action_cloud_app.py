"""Driver app for RunActionScreen snapshot test (cloud-only run)."""

from textual.app import App

from training.run_manager import RunInfo, RunStatus
from tui.screens.run_action import RunActionScreen

MOCK_RUN_INFO = RunInfo(
    name="s2w-lstm-0228-0930",
    bot_name="simple2wheeler",
    policy_type="lstm",
    algorithm="ppo",
    scene_path="",
    status=RunStatus.RUNNING,
    created_at="2026-02-28T09:30:00",
    wandb_id="def456",
    wandb_url="https://wandb.ai/mindsim/s2w-lstm-0228-0930",
)


class RunActionCloudApp(App):
    def on_mount(self) -> None:
        self.push_screen(
            RunActionScreen(
                run_dir=None,
                run_info=MOCK_RUN_INFO,
                cloud_only=True,
            )
        )


if __name__ == "__main__":
    RunActionCloudApp().run()
