"""Run browser screen for MindSim TUI."""

from __future__ import annotations

import webbrowser
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static

from training.run_manager import (
    BOT_DISPLAY_NAMES,
    RunInfo,
    RunStatus,
    bot_display_name,
    discover_local_runs,
    discover_wandb_runs,
)


class RunBrowserScreen(Screen):
    """Browse local and W&B runs.

    If ``runs`` is provided, uses that data directly (useful for testing).
    Otherwise discovers runs from the filesystem and W&B API.
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back", priority=True),
        Binding("backspace", "go_back", "Back", show=False, priority=True),
        Binding("enter", "select_run", "Select", priority=True),
        Binding("w", "open_wandb", "W&B", priority=True),
    ]

    CSS = """
    RunBrowserScreen {
        layout: vertical;
    }

    #browser-box {
        width: 1fr;
        max-width: 120;
        height: 1fr;
        border: ascii $accent;
        padding: 1 2;
        margin: 1 2;
    }

    #browser-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #run-list {
        height: 1fr;
    }

    #loading-status {
        color: $text-muted;
    }

    #no-runs {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        runs: list[tuple[Path | None, RunInfo]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._provided_runs = runs  # None = discover at compose time
        self._items: list[dict] = []  # maps list index -> run info

    def _build_items(self) -> list[str]:
        """Merge local and W&B runs into a single list, deduped by W&B id.

        Returns labels sorted by created_at descending (newest first).
        """
        # Collect all entries as (sort_key, label, item_dict) then sort
        entries: list[tuple[str, str, dict]] = []

        if self._provided_runs is not None:
            # Use injected data — all treated as local-style entries
            for run_dir, info in self._provided_runs:
                date_str = info.created_at[:10] if info.created_at else "?"
                display_name = bot_display_name(info.bot_name)
                status_str = RunStatus.label(info.status)
                batch_str = f"b{info.batch_idx}" if info.batch_idx else ""
                source = "LOCAL" if run_dir is not None else "CLOUD"
                label = f"{source}  {status_str}  {info.name}  {display_name}  {date_str}  {batch_str}"
                sort_key = info.created_at or ""
                cloud_only = run_dir is None
                entries.append(
                    (
                        sort_key,
                        label,
                        {
                            "type": "cloud" if cloud_only else "local",
                            "dir": run_dir,
                            "info": info,
                        },
                    )
                )
        else:
            # Discover from filesystem + W&B
            local_wandb_ids: set[str] = set()
            for run_dir, info in discover_local_runs():
                date_str = info.created_at[:10] if info.created_at else "?"
                display_name = bot_display_name(info.bot_name)
                status_str = RunStatus.label(info.status)
                batch_str = f"b{info.batch_idx}" if info.batch_idx else ""
                label = f"LOCAL  {status_str}  {info.name}  {display_name}  {date_str}  {batch_str}"
                sort_key = info.created_at or ""
                entries.append(
                    (
                        sort_key,
                        label,
                        {"type": "local", "dir": run_dir, "info": info},
                    )
                )
                if info.wandb_id:
                    local_wandb_ids.add(info.wandb_id)

            # W&B runs — skip any already shown as local
            for wr in discover_wandb_runs(limit=50):
                if wr["id"] in local_wandb_ids:
                    continue
                date_str = wr["created_at"][:10] if wr.get("created_at") else "?"
                state_str = RunStatus.label(wr.get("state", ""))
                tags_str = " ".join(wr.get("tags", []))
                label = f"CLOUD  {state_str}  {wr['name']}  {tags_str}  {date_str}"
                bot_name = ""
                policy_type = ""
                tags = wr.get("tags", [])
                for t in tags:
                    if t in BOT_DISPLAY_NAMES:
                        bot_name = t
                    elif t in ("lstm", "mlp", "cnn"):
                        policy_type = t
                cloud_info = RunInfo(
                    name=wr["name"],
                    bot_name=bot_name,
                    policy_type=policy_type,
                    algorithm=next(
                        (t for t in tags if t not in (bot_name, policy_type)), ""
                    ),
                    scene_path=f"bots/{bot_name}/scene.xml" if bot_name else "",
                    status=wr.get("state", "unknown"),
                    wandb_id=wr["id"],
                    wandb_url=wr.get("url"),
                    created_at=wr.get("created_at"),
                    tags=tags,
                )
                sort_key = wr.get("created_at", "")
                entries.append((sort_key, label, {"type": "cloud", "info": cloud_info}))

        # Sort newest first
        entries.sort(key=lambda e: e[0], reverse=True)

        labels = []
        for _, label, item in entries:
            labels.append(label)
            self._items.append(item)
        return labels

    def compose(self) -> ComposeResult:
        with Vertical(id="browser-box"):
            yield Static("Browse Runs", id="browser-title")
            if self._provided_runs is not None:
                # Sync path: injected data (used by tests)
                labels = self._build_items()
                if labels:
                    yield OptionList(*labels, id="run-list")
                else:
                    yield Static("  No runs found.", id="no-runs")
            else:
                # Async path: show loading placeholder, discover in worker
                yield Static("  Loading...", id="loading-status")
        yield Footer()

    def on_mount(self) -> None:
        if self._provided_runs is None:
            self._load_runs()

    @work(thread=True)
    def _load_runs(self) -> None:
        labels = self._build_items()
        self.app.call_from_thread(self._populate_list, labels)

    def _populate_list(self, labels: list[str]) -> None:
        """Replace loading placeholder with the run list (called on main thread)."""
        # Remove loading indicator
        try:
            loading = self.query_one("#loading-status", Static)
            loading.remove()
        except Exception:
            pass

        box = self.query_one("#browser-box", Vertical)
        if labels:
            box.mount(OptionList(*labels, id="run-list"))
        else:
            box.mount(Static("  No runs found.", id="no-runs"))

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_select_run(self) -> None:
        try:
            run_list = self.query_one("#run-list", OptionList)
            idx = run_list.highlighted
            if idx is not None and 0 <= idx < len(self._items):
                item = self._items[idx]
                run_dir = item.get("dir")
                from tui.screens.run_action import RunActionScreen

                self.app.push_screen(
                    RunActionScreen(
                        run_dir=run_dir,
                        run_info=item["info"],
                        cloud_only=run_dir is None,
                    )
                )
        except (IndexError, ValueError):
            pass

    def action_open_wandb(self) -> None:
        try:
            run_list = self.query_one("#run-list", OptionList)
            idx = run_list.highlighted
            if idx is not None and 0 <= idx < len(self._items):
                url = self._items[idx]["info"].wandb_url
                if url:
                    webbrowser.open(url)
        except (IndexError, ValueError):
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_select_run()
