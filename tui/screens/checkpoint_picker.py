"""Checkpoint picker screen for MindSim TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static


class CheckpointPickerScreen(Screen):
    """Pick a checkpoint from a run before playing or resuming."""

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "confirm", "Select", priority=True),
    ]

    CSS = """
    CheckpointPickerScreen {
        align: center middle;
    }

    #picker-box {
        width: 70;
        height: auto;
        max-height: 30;
        border: ascii $accent;
        padding: 1 2;
    }

    #picker-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #checkpoint-list {
        height: auto;
        max-height: 22;
    }

    #no-checkpoints {
        color: $text-muted;
    }
    """

    def __init__(self, run_dir: Path, run_info, mode: str, **kwargs):
        """
        Args:
            run_dir: Path to the run directory.
            run_info: RunInfo for the selected run.
            mode: "play" or "resume".
        """
        super().__init__(**kwargs)
        self._run_dir = run_dir
        self._info = run_info
        self._mode = mode
        from training.checkpoint import list_checkpoints

        self._checkpoints = list_checkpoints(run_dir)

    def compose(self) -> ComposeResult:
        action_label = "Play" if self._mode == "play" else "Resume"
        with Vertical(id="picker-box"):
            yield Static(f"{action_label}: {self._info.name}", id="picker-title")
            if self._checkpoints:
                items = []
                for i, ckpt in enumerate(self._checkpoints):
                    stage = (
                        f"Stage {ckpt['stage']}" if ckpt["stage"] is not None else "?"
                    )
                    batch = (
                        f"Batch {ckpt['batch']}" if ckpt["batch"] is not None else "?"
                    )
                    tag = "  (latest)" if i == 0 else ""
                    items.append(f"{stage}  {batch}{tag}")
                yield OptionList(*items, id="checkpoint-list")
            else:
                yield Static("  No checkpoints found.", id="no-checkpoints")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_confirm(self) -> None:
        if not self._checkpoints:
            self.app.pop_screen()
            return

        try:
            ol = self.query_one("#checkpoint-list", OptionList)
            idx = ol.highlighted
            if idx is None:
                idx = 0
        except Exception:
            idx = 0

        ckpt_path = self._checkpoints[idx]["path"]

        if self._mode == "play":
            self.app.start_playing_run(
                run_name=self._info.name,
                scene_path=self._info.scene_path,
                checkpoint_path=ckpt_path,
            )
        else:
            self.app.start_training(
                smoketest=False,
                scene_path=self._info.scene_path,
                resume=ckpt_path,
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_confirm()
