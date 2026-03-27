"""Bot selector screen for MindSim TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, RadioButton, RadioSet, Static

from training.run_manager import bot_display_name


class BotSelectorScreen(Screen):
    """Select a bot for training or viewing."""

    BINDINGS = [  # noqa: RUF012
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "confirm", "Select", priority=True),
    ]

    CSS = """
    BotSelectorScreen {
        align: center middle;
    }

    #bot-box {
        width: 50;
        height: auto;
        border: ascii $accent;
        padding: 1 2;
    }

    #bot-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #bot-selector {
        height: auto;
    }
    """

    def __init__(self, bots: list[dict], mode: str = "train", **kwargs):
        super().__init__(**kwargs)
        self._bots = bots
        self._mode = mode

    def compose(self) -> ComposeResult:
        title = "Select Bot to View" if self._mode == "view" else "Select Bot"
        with Vertical(id="bot-box"):
            yield Static(title, id="bot-title")
            if self._bots:
                with RadioSet(id="bot-selector"):
                    for i, bot in enumerate(self._bots):
                        display = bot_display_name(bot["name"])
                        yield RadioButton(f"{display} ({bot['name']})", value=(i == 0))
            else:
                yield Static("  No bots found in bots/*/scene.xml")
        yield Footer()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def _get_selected_scene(self) -> str | None:
        if not self._bots:
            return None
        try:
            radio_set = self.query_one("#bot-selector", RadioSet)
            idx = radio_set.pressed_index
            if idx >= 0:
                return self._bots[idx]["scene_path"]
        except (IndexError, ValueError):
            pass
        return self._bots[0]["scene_path"]

    def action_confirm(self) -> None:
        scene_path = self._get_selected_scene()
        if not scene_path:
            return
        if self._mode == "view":
            self.app.start_viewing(scene_path=scene_path)
        else:
            self.app.start_training(smoketest=False, scene_path=scene_path)
