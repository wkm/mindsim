"""Main menu screen for MindSim TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static


class MainMenuScreen(Screen):
    """Top-level menu: smoketest, new run, browse runs, quit."""

    BINDINGS = [
        Binding("s", "select('smoketest')", "Smoketest", priority=True),
        Binding("n", "select('new')", "New Run", priority=True),
        Binding("g", "select('scene')", "Scene Gen", priority=True),
        Binding("v", "select('view')", "View Bot", priority=True),
        Binding("b", "select('browse')", "Browse Runs", priority=True),
        Binding("c", "select('gcp')", "GCP Instances", priority=True),
        Binding("q", "select('quit')", "Quit", priority=True),
        Binding("escape", "select('quit')", "Quit", show=False, priority=True),
        Binding("backspace", "select('quit')", "Quit", show=False, priority=True),
    ]

    CSS = """
    MainMenuScreen {
        align: center middle;
    }

    #menu-box {
        width: 50;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #menu-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }

    #menu-list {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-box"):
            yield Static("MindSim", id="menu-title")
            yield OptionList(
                "\\[s] Smoketest",
                "\\[n] New training run",
                "\\[g] Scene gen preview",
                "\\[v] View bot",
                "\\[b] Browse runs",
                "\\[c] GCP instances",
                "\\[q] Quit",
                id="menu-list",
            )
        yield Footer()

    def action_select(self, choice: str) -> None:
        if choice == "smoketest":
            self.app.start_training(smoketest=True)
        elif choice == "new":
            from main import _discover_bots
            from tui.screens.bot_selector import BotSelectorScreen

            self.app.push_screen(BotSelectorScreen(bots=_discover_bots(), mode="train"))
        elif choice == "scene":
            self.app.start_scene_preview()
        elif choice == "view":
            from main import _discover_bots
            from tui.screens.bot_selector import BotSelectorScreen

            self.app.push_screen(BotSelectorScreen(bots=_discover_bots(), mode="view"))
        elif choice == "browse":
            from tui.screens.run_browser import RunBrowserScreen

            self.app.push_screen(RunBrowserScreen())
        elif choice == "gcp":
            from tui.screens.gcp_instances import GCPInstancesScreen

            self.app.push_screen(GCPInstancesScreen())
        elif choice == "quit":
            self.app.exit()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        choices = ["smoketest", "new", "scene", "view", "browse", "gcp", "quit"]
        if 0 <= idx < len(choices):
            self.action_select(choices[idx])
