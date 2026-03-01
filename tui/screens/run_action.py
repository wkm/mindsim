"""Run action screen for MindSim TUI."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from textual import work
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

    #download-status {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(
        self, run_dir: Path | None, run_info, cloud_only: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self._run_dir = run_dir
        self._info = run_info
        self._cloud_only = cloud_only
        self._downloading = False

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
        if not self._downloading:
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
        if self._downloading:
            return
        self._downloading = True
        self._start_download()

    @work(thread=True)
    def _start_download(self) -> None:
        from replay import _open_in_rerun, discover_recordings

        run_name = self._info.name

        self.app.call_from_thread(
            self._show_download_status, "Discovering recordings..."
        )

        try:
            available = discover_recordings(run_name)
        except Exception as e:
            self.app.call_from_thread(
                self._show_download_status, f"Error discovering recordings: {e}"
            )
            self.app.call_from_thread(self._finish_download)
            return

        if not available:
            self.app.call_from_thread(
                self._show_download_status,
                f"No recordings found for '{run_name}'.",
            )
            self.app.call_from_thread(self._finish_download)
            return

        # Take last 3 recordings
        selected = available[-3:]

        self.app.call_from_thread(
            self._show_download_status,
            f"Found {len(available)} recordings. Downloading {len(selected)}...",
        )

        output_dir = Path(f"replay_{run_name}")
        output_dir.mkdir(exist_ok=True)

        rrd_paths = []
        for i, rec in enumerate(selected):
            label = (
                f"batch {rec.batch_idx}"
                if rec.batch_idx is not None
                else f"episode {rec.episode}"
            )
            self.app.call_from_thread(
                self._show_download_status,
                f"[{i + 1}/{len(selected)}] Downloading {label}...",
            )
            try:
                # Download artifact from W&B if needed
                if rec.artifact is not None:
                    artifact_dir = rec.artifact.download()
                    rrd_files = list(Path(artifact_dir).glob("*.rrd"))
                    if not rrd_files:
                        raise FileNotFoundError(f"No .rrd in artifact for {label}")
                    src = rrd_files[0]
                    if rec.batch_idx is not None:
                        dest = output_dir / f"batch_{rec.batch_idx}.rrd"
                    else:
                        dest = output_dir / f"episode_{rec.episode}.rrd"
                    # Remove stale symlinks before creating new ones
                    if dest.is_symlink():
                        dest.unlink()
                    if not dest.exists():
                        dest.symlink_to(src.resolve())
                    rrd_paths.append(str(dest))
                elif rec.path and Path(rec.path).exists():
                    rrd_paths.append(rec.path)
                else:
                    raise FileNotFoundError(f"No artifact or local path for {label}")
            except Exception as e:
                self.app.call_from_thread(
                    self._show_download_status,
                    f"[{i + 1}/{len(selected)}] {label} — error: {e}",
                )

        if rrd_paths:
            self.app.call_from_thread(
                self._show_download_status,
                f"Done — {len(rrd_paths)} recordings in {output_dir}/. Opening Rerun...",
            )
            _open_in_rerun(rrd_paths)
        else:
            self.app.call_from_thread(
                self._show_download_status, "No recordings downloaded."
            )

        self.app.call_from_thread(self._finish_download)

    def _show_download_status(self, text: str) -> None:
        """Update or create the download status widget."""
        try:
            status = self.query_one("#download-status", Static)
            status.update(text)
        except Exception:
            box = self.query_one("#action-box", Vertical)
            box.mount(Static(text, id="download-status"))

    def _finish_download(self) -> None:
        self._downloading = False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if 0 <= idx < len(self._action_map):
            getattr(self, f"action_{self._action_map[idx]}")()
