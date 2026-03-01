"""Dirty tree warning screen for MindSim TUI."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static


class DirtyTreeScreen(Screen):
    """Shown before training when the git worktree has uncommitted changes."""

    BINDINGS = [
        Binding("c", "commit", "Commit with Claude", priority=True),
        Binding("s", "start_anyway", "Start Anyway", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
    ]

    CSS = """
    DirtyTreeScreen {
        align: center middle;
    }

    #dirty-box {
        width: 60;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #dirty-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dirty-status {
        margin-bottom: 1;
    }

    #dirty-actions {
        height: auto;
    }
    """

    def __init__(
        self,
        status_lines: str,
        smoketest: bool,
        scene_path: str | None,
        resume: str | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._status_lines = status_lines
        self._smoketest = smoketest
        self._scene_path = scene_path
        self._resume = resume

    def compose(self) -> ComposeResult:
        with Vertical(id="dirty-box"):
            yield Static("Uncommitted Changes", id="dirty-title")
            yield Static(self._status_lines, id="dirty-status")
            yield OptionList(
                "[c] Commit with Claude",
                "[s] Start anyway",
                "[Esc] Back",
                id="dirty-actions",
            )
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_start_anyway(self) -> None:
        self.app.pop_screen()
        self.app._do_start_training(
            smoketest=self._smoketest,
            scene_path=self._scene_path,
            resume=self._resume,
        )

    def action_commit(self) -> None:
        self.query_one("#dirty-title", Static).update("Committing...")
        self._do_commit()

    @work(thread=True)
    def _do_commit(self) -> None:
        from main import _git_is_clean, _run_claude_commit

        clean = _run_claude_commit()
        if clean:
            self.app.call_from_thread(self._start_after_commit)
        else:
            _, new_status = _git_is_clean()
            self.app.call_from_thread(self._update_after_commit, new_status)

    def _start_after_commit(self) -> None:
        self.app.pop_screen()
        self.app._do_start_training(
            smoketest=self._smoketest,
            scene_path=self._scene_path,
            resume=self._resume,
        )

    def _update_after_commit(self, new_status: str) -> None:
        self.query_one("#dirty-title", Static).update("Still Uncommitted Changes")
        self.query_one("#dirty-status", Static).update(new_status)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        actions = ["commit", "start_anyway", "go_back"]
        if 0 <= idx < len(actions):
            getattr(self, f"action_{actions[idx]}")()
