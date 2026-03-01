"""Driver app for RunBrowserScreen snapshot test with mock data."""

from pathlib import Path

from textual.app import App

from training.run_manager import RunInfo
from tui.screens.run_browser import RunBrowserScreen

MOCK_RUNS = [
    (
        Path("runs/s2w-lstm-0301-1045"),
        RunInfo(
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
        ),
    ),
    (
        None,
        RunInfo(
            name="s2w-lstm-0228-0930",
            bot_name="simple2wheeler",
            policy_type="lstm",
            algorithm="ppo",
            scene_path="",
            status="running",
            created_at="2026-02-28T09:30:00",
            wandb_id="def456",
        ),
    ),
    (
        Path("runs/biped-mlp-0225-1400"),
        RunInfo(
            name="biped-mlp-0225-1400",
            bot_name="simplebiped",
            policy_type="mlp",
            algorithm="ppo",
            scene_path="bots/simplebiped/scene.xml",
            status="failed",
            batch_idx=200,
            episode_count=500,
            curriculum_stage=1,
            created_at="2026-02-25T14:00:00",
        ),
    ),
]


class RunBrowserApp(App):
    def on_mount(self) -> None:
        self.push_screen(RunBrowserScreen(runs=MOCK_RUNS))


if __name__ == "__main__":
    RunBrowserApp().run()
