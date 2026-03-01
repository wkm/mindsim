"""Run action screen for MindSim TUI."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static

from run_manager import bot_display_name


class RunActionScreen(Screen):
    """Actions for a selected run: play, resume, view, W&B, back."""

    BINDINGS = [
        Binding("p", "play_run", "Play", priority=True),
        Binding("r", "resume_run", "Resume", priority=True),
        Binding("v", "view_run", "View", priority=True),
        Binding("d", "download_recordings", "Recordings", priority=True),
        Binding("w", "open_wandb", "W&B", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
    ]

    CSS = """
    RunActionScreen {
        align: center middle;
    }

    #action-box {
        width: 60;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #action-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #run-metadata {
        color: $text-muted;
        margin-bottom: 1;
    }

    #action-list {
        height: auto;
    }
    """

    def __init__(
        self, run_dir: Path | None, run_info, cloud_only: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self._run_dir = run_dir
        self._info = run_info
        self._cloud_only = cloud_only

    def compose(self) -> ComposeResult:
        info = self._info
        display_name = bot_display_name(info.bot_name) if info.bot_name else "?"
        source = "CLOUD" if self._cloud_only else "LOCAL"
        with Vertical(id="action-box"):
            yield Static(f"Run: {info.name}  ({source})", id="action-title")
            meta_lines = [
                f"  Bot: {display_name} ({info.bot_name})",
                f"  Algorithm: {info.algorithm}  |  Policy: {info.policy_type}",
                f"  Status: {info.status}  |  Created: {info.created_at or '?'}",
            ]
            if not self._cloud_only:
                meta_lines.insert(
                    2,
                    f"  Batches: {info.batch_idx}  |  Episodes: {info.episode_count}  |  Stage: {info.curriculum_stage}",
                )
            if info.wandb_url:
                meta_lines.append(f"  W&B: {info.wandb_url}")
            yield Static("\n".join(meta_lines), id="run-metadata")
            options = []
            self._action_map = []
            if not self._cloud_only:
                options += [
                    "[p] Play checkpoint",
                    "[r] Resume training",
                    "[v] View in MuJoCo",
                ]
                self._action_map += ["play_run", "resume_run", "view_run"]
            options.append("[d] Download recordings")
            self._action_map.append("download_recordings")
            if info.wandb_url:
                options.append("[w] Open W&B")
                self._action_map.append("open_wandb")
            options.append("[Esc] Back")
            self._action_map.append("go_back")
            yield OptionList(*options, id="action-list")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_play_run(self) -> None:
        from main import CheckpointPickerScreen

        self.app.push_screen(
            CheckpointPickerScreen(
                run_dir=self._run_dir,
                run_info=self._info,
                mode="play",
            )
        )

    def action_resume_run(self) -> None:
        from main import CheckpointPickerScreen

        self.app.push_screen(
            CheckpointPickerScreen(
                run_dir=self._run_dir,
                run_info=self._info,
                mode="resume",
            )
        )

    def action_view_run(self) -> None:
        self.app.start_viewing(scene_path=self._info.scene_path)

    def action_open_wandb(self) -> None:
        url = self._info.wandb_url
        if url:
            webbrowser.open(url)

    def action_download_recordings(self) -> None:
        from replay import run_download

        run_name = self._info.name
        self.app.exit()
        run_download(run_name, last_n=3)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if 0 <= idx < len(self._action_map):
            getattr(self, f"action_{self._action_map[idx]}")()
