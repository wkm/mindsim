"""GCP instances management screen for MindSim TUI."""

from __future__ import annotations

import json
import subprocess
import time
import webbrowser

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static


class GCPInstancesScreen(Screen):
    """List and manage GCP mindsim-* instances."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("w", "open_wandb", "W&B", priority=True),
        Binding("t", "terminate", "Terminate", priority=True),
        Binding("d", "delete", "Delete", priority=True),
        Binding("escape", "go_back", "Back", priority=True),
    ]

    CSS = """
    GCPInstancesScreen {
        layout: vertical;
    }

    #gcp-title {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }

    #gcp-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    #gcp-table {
        height: 1fr;
        margin: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._confirm_action: str | None = None  # "delete" or "terminate"
        self._confirm_name: str | None = None
        self._confirm_zone: str | None = None
        self._confirm_time: float = 0.0

    def compose(self) -> ComposeResult:
        yield Static("GCP Instances", id="gcp-title")
        yield Static("Loading...", id="gcp-status")
        table = DataTable(id="gcp-table")
        table.cursor_type = "row"
        table.add_columns(
            "Name", "Status", "Run", "Branch", "Args", "Created", "Zone", "Machine Type"
        )
        yield table
        yield Footer()

    def on_mount(self) -> None:
        self._load_instances()

    def action_refresh(self) -> None:
        self._clear_confirm()
        self.query_one("#gcp-status", Static).update("Refreshing...")
        self._load_instances()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_delete(self) -> None:
        self._confirm_or_execute("delete", "d")

    def action_terminate(self) -> None:
        self._confirm_or_execute("terminate", "t")

    def action_open_wandb(self) -> None:
        """Open the W&B run page for the selected instance."""
        table = self.query_one("#gcp-table", DataTable)
        if table.row_count == 0:
            return
        row = table.get_row_at(table.cursor_row)
        name = row[0]  # Instance name = W&B run name
        args = row[4]  # "Args" column
        if not name:
            return
        project = "mindsim-biped" if "biped" in args else "mindsim-2wheeler"
        url = f"https://wandb.ai/search?q={name}&project={project}"
        webbrowser.open(url)
        self.query_one("#gcp-status", Static).update(f"Opened W&B for {name}")

    def _confirm_or_execute(self, action: str, key: str) -> None:
        table = self.query_one("#gcp-table", DataTable)
        if table.row_count == 0:
            return

        row_key = table.cursor_row
        row = table.get_row_at(row_key)
        name, zone = row[0], row[6]

        now = time.monotonic()
        if (
            self._confirm_action == action
            and self._confirm_name == name
            and (now - self._confirm_time) < 5.0
        ):
            # Second press — execute
            self._clear_confirm()
            self.query_one("#gcp-status", Static).update(
                f"{'Deleting' if action == 'delete' else 'Terminating'} {name}..."
            )
            if action == "delete":
                self._delete_instance(name, zone, row_key)
            else:
                self._stop_instance(name, zone)
        else:
            # First press — ask for confirmation
            self._confirm_action = action
            self._confirm_name = name
            self._confirm_zone = zone
            self._confirm_time = now
            self.query_one("#gcp-status", Static).update(
                f"{action.capitalize()} {name}? Press {key} again to confirm"
            )

    def _clear_confirm(self) -> None:
        self._confirm_action = None
        self._confirm_name = None
        self._confirm_zone = None
        self._confirm_time = 0.0

    @work(thread=True)
    def _load_instances(self) -> None:
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "list",
                    "--filter=labels.mindsim=true",
                    "--format=json(name,zone.basename(),status,machineType.basename(),labels,creationTimestamp)",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                self.app.call_from_thread(
                    self._set_status, f"Error: {result.stderr.strip()}"
                )
                return
            instances = json.loads(result.stdout)
        except FileNotFoundError:
            self.app.call_from_thread(self._set_status, "gcloud CLI not found")
            return
        except Exception as e:
            self.app.call_from_thread(self._set_status, f"Error: {e}")
            return

        self.app.call_from_thread(self._populate_table, instances)

    def _set_status(self, text: str) -> None:
        self.query_one("#gcp-status", Static).update(text)

    def _populate_table(self, instances: list[dict]) -> None:
        table = self.query_one("#gcp-table", DataTable)
        table.clear()

        for inst in instances:
            name = inst.get("name", "")
            zone = inst.get("zone", "")
            status = inst.get("status", "")
            machine = inst.get("machineType", "")
            labels = inst.get("labels", {}) or {}
            run = labels.get("mindsim-run", "")
            branch = labels.get("mindsim-branch", "")
            args = labels.get("mindsim-args", "")
            created = inst.get("creationTimestamp", "")
            # Shorten timestamp: "2026-02-14T16:34:56.123-08:00" → "2026-02-14 16:34"
            if "T" in created:
                created = created.split("T")[0] + " " + created.split("T")[1][:5]

            # Color-code status
            from botcad.colors import TUI_ERROR, TUI_SUCCESS, TUI_WARNING

            if status == "RUNNING":
                status_display = f"[{TUI_SUCCESS}]{status}[/{TUI_SUCCESS}]"
            elif status in ("TERMINATED", "STOPPED"):
                status_display = f"[{TUI_ERROR}]{status}[/{TUI_ERROR}]"
            elif status in ("STAGING", "PROVISIONING", "SUSPENDING"):
                status_display = f"[{TUI_WARNING}]{status}[/{TUI_WARNING}]"
            else:
                status_display = status

            table.add_row(
                name, status_display, run, branch, args, created, zone, machine
            )

        count = len(instances)
        self.query_one("#gcp-status", Static).update(
            f"{count} instance{'s' if count != 1 else ''}"
        )

    @work(thread=True)
    def _delete_instance(self, name: str, zone: str, row_index: int) -> None:
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "delete",
                    name,
                    f"--zone={zone}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                self.app.call_from_thread(self._on_delete_success, name, row_index)
            else:
                self.app.call_from_thread(
                    self._set_status, f"Delete failed: {result.stderr.strip()}"
                )
        except Exception as e:
            self.app.call_from_thread(self._set_status, f"Delete error: {e}")

    def _on_delete_success(self, name: str, row_index: int) -> None:
        self._set_status(f"Deleted {name}. Refreshing...")
        self._load_instances()

    @work(thread=True)
    def _stop_instance(self, name: str, zone: str) -> None:
        try:
            result = subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "stop",
                    name,
                    f"--zone={zone}",
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                self.app.call_from_thread(self._on_stop_success, name)
            else:
                self.app.call_from_thread(
                    self._set_status, f"Terminate failed: {result.stderr.strip()}"
                )
        except Exception as e:
            self.app.call_from_thread(self._set_status, f"Terminate error: {e}")

    def _on_stop_success(self, name: str) -> None:
        self._set_status(f"Terminated {name}. Refreshing...")
        self._load_instances()
