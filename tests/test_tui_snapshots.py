"""Snapshot tests for TUI screens.

Uses pytest-textual-snapshot to render each screen headlessly and compare
SVG output against saved baselines. First run generates baselines; subsequent
runs diff against them.

Regenerate baselines after intentional UI changes:
    uv run pytest tests/test_tui_snapshots.py --snapshot-update
"""

from pathlib import Path

DRIVERS = Path(__file__).parent / "snapshot_drivers"


def test_main_menu(snap_compare):
    assert snap_compare(str(DRIVERS / "main_menu_app.py"))


def test_run_browser(snap_compare):
    assert snap_compare(str(DRIVERS / "run_browser_app.py"))


def test_run_action_local(snap_compare):
    assert snap_compare(str(DRIVERS / "run_action_local_app.py"))


def test_run_action_cloud(snap_compare):
    assert snap_compare(str(DRIVERS / "run_action_cloud_app.py"))
