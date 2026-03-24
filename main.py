"""
Textual TUI for MindSim training.

Single entry point for all MindSim modes: interactive TUI, view, play, train,
smoketest, quicksim, and visualize.

Usage:
    uv run mjpython main.py                    # Interactive TUI (default)
    uv run mjpython main.py view [--bot NAME]  # MuJoCo viewer
    uv run mjpython main.py play [CHECKPOINT] [--bot NAME]  # Play trained policy
    uv run mjpython main.py train [--smoketest] [--bot NAME] [--resume REF] [--num-workers N]
    uv run mjpython main.py smoketest          # Alias for train --smoketest
    uv run mjpython main.py quicksim           # Rerun debug vis
    uv run mjpython main.py visualize [--bot NAME] [--steps N]

Requires mjpython (not plain python) for MuJoCo viewer/play features.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.theme import Theme

from botcad.colors import (
    TUI_BACKGROUND,
    TUI_ERROR,
    TUI_PRIMARY,
    TUI_PRIMARY_BG,
    TUI_SECONDARY,
    TUI_SUCCESS,
    TUI_SURFACE,
    TUI_WARNING,
)
from training.train import CommandChannel
from tui.screens.dirty_tree import DirtyTreeScreen
from tui.screens.main_menu import MainMenuScreen
from tui.screens.training_dashboard import TrainingDashboard

# Blueprint.js dark theme for Textual TUI
_BLUEPRINT_THEME = Theme(
    name="blueprint",
    primary=TUI_PRIMARY,
    secondary=TUI_SECONDARY,
    warning=TUI_WARNING,
    error=TUI_ERROR,
    success=TUI_SUCCESS,
    background=TUI_BACKGROUND,
    surface=TUI_SURFACE,
    panel=TUI_PRIMARY_BG,
    dark=True,
)

log = logging.getLogger(__name__)


class TuiLogHandler(logging.Handler):
    """Routes Python log records into the TUI log panel.

    This is the *only* path for messages to reach the log panel from
    worker threads.  ``dashboard.message()`` just calls ``log.info()``
    and this handler forwards the record to the RichLog widget.

    UI actions on the event-loop thread write to the log panel directly
    (calling ``call_from_thread`` from the event loop would deadlock),
    so we skip records originating on that thread.

    Installed when training starts, removed when training finishes
    to avoid stale references to a dead app.
    """

    def __init__(self, app: MindSimApp):
        super().__init__(level=logging.INFO)
        self._app = app
        self._event_loop_thread = threading.current_thread()
        fmt = logging.Formatter("%(message)s")
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord):
        # UI actions on the event-loop thread write to the panel directly;
        # routing them through call_from_thread would deadlock.
        if threading.current_thread() is self._event_loop_thread:
            return
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                msg = f"[bold red]{msg}[/bold red]"
            elif record.levelno >= logging.WARNING:
                msg = f"[bold yellow]{msg}[/bold yellow]"
            self._app.call_from_thread(self._app.log_message, msg)
        except Exception:
            pass  # Don't let logging errors crash the app


def _discover_bots() -> list[dict]:
    """Scan bots/*/scene.xml and return info about each bot."""
    bots_dir = Path("bots")
    results = []
    if bots_dir.is_dir():
        for scene in sorted(bots_dir.glob("*/scene.xml")):
            name = scene.parent.name
            results.append({"name": name, "scene_path": str(scene)})
    return results


def _resolve_scene_path(bot_name: str | None) -> str:
    """Resolve a bot name to its scene.xml path.

    Args:
        bot_name: Bot directory name (e.g. "simplebiped", "simple2wheeler"),
                  or None for the default bot.

    Returns:
        Path to the bot's scene.xml file.

    Raises:
        SystemExit: If the bot name is not found.
    """
    bots = _discover_bots()
    if not bots:
        print("Error: No bots found in bots/*/scene.xml", file=sys.stderr)
        sys.exit(1)

    if bot_name is None:
        # Default to first bot (simple2wheeler comes before simplebiped alphabetically)
        return bots[0]["scene_path"]

    for bot in bots:
        if bot["name"] == bot_name:
            return bot["scene_path"]

    available = ", ".join(b["name"] for b in bots)
    print(f"Error: Unknown bot '{bot_name}'. Available: {available}", file=sys.stderr)
    sys.exit(1)


def _get_experiment_info(branch: str) -> str | None:
    """Look up the hypothesis for a branch from EXPERIMENTS.md.

    Parses the markdown table and returns the hypothesis text for the
    matching branch, or None if not found.
    """
    experiments_path = Path("EXPERIMENTS.md")
    if not experiments_path.exists():
        return None
    try:
        text = experiments_path.read_text()
    except OSError:
        return None

    for line in text.splitlines():
        # Match table rows: | `branch` | hypothesis | ... |
        m = re.match(r"\|\s*`([^`]+)`\s*\|([^|]+)\|", line)
        if m and m.group(1).strip() == branch:
            return m.group(2).strip()
    return None


def _git_is_clean() -> tuple[bool, str]:
    """Check whether the git working tree is clean.

    Returns (is_clean, status_lines) where status_lines is the raw
    ``git status --porcelain`` output.  If this isn't a git repo or
    git isn't available, returns (True, "") so training proceeds.
    """

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        return (output == "", output)
    except Exception:
        return (True, "")


def _run_claude_commit() -> bool:
    """Run Claude in print mode to commit all changes.

    Returns True if the worktree is clean afterward.
    """

    print("Running Claude to commit changes...")
    try:
        subprocess.run(
            [
                "claude",
                "-p",
                "Run git status and git diff to see current changes, then commit all changes with a descriptive commit message.",
                "--allowedTools",
                "Bash Read Grep Glob",
            ],
            timeout=120,
        )
    except Exception as e:
        print(f"Claude commit failed: {e}")
        return False

    clean, _ = _git_is_clean()
    return clean


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


class MindSimApp(App):
    """MindSim Textual TUI application."""

    TITLE = "MindSim"
    CSS = """
    Screen {
        background: $surface;
    }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_theme(_BLUEPRINT_THEME)
        self.theme = "blueprint"
        self.commands = CommandChannel()
        self._dashboard: TrainingDashboard | None = None
        # Set by screens to dispatch after app.run() returns
        self.next_action: str | None = None
        self.next_scene: str | None = None
        self.next_stage: int | None = None
        self.next_run_name: str | None = None
        self.next_checkpoint_path: str | None = None

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())

    def start_viewing(self, scene_path: str | None = None, stage: int | None = None):
        """Exit TUI, then main() will launch the MuJoCo viewer."""
        if not scene_path:
            return
        self.next_action = "view"
        self.next_scene = scene_path
        self.next_stage = stage
        self.exit()

    def start_scene_preview(self):
        """Exit TUI, then main() will launch the scene preview."""
        self.next_action = "scene"
        self.exit()

    def start_playing(self, scene_path: str | None = None):
        """Exit TUI, then main() will launch play mode."""
        self.next_action = "play"
        self.next_scene = scene_path
        self.exit()

    def start_playing_run(
        self,
        run_name: str | None = None,
        scene_path: str | None = None,
        checkpoint_path: str | None = None,
    ):
        """Exit TUI, then main() will launch play mode for a specific run."""
        self.next_action = "play"
        self.next_scene = scene_path
        self.next_run_name = run_name
        self.next_checkpoint_path = checkpoint_path
        self.exit()

    def start_training(
        self,
        smoketest: bool = False,
        scene_path: str | None = None,
        resume: str | None = None,
    ):
        """Called by screens to start training (with dirty-tree gate)."""
        # If no scene_path provided (e.g. smoketest from main menu), use default
        if scene_path is None:
            bots = _discover_bots()
            scene_path = bots[0]["scene_path"] if bots else None

        # Smoketests skip the dirty-tree check
        if smoketest:
            self._do_start_training(
                smoketest=True, scene_path=scene_path, resume=resume
            )
            return

        clean, status = _git_is_clean()
        if clean:
            self._do_start_training(
                smoketest=False, scene_path=scene_path, resume=resume
            )
        else:
            self.push_screen(
                DirtyTreeScreen(
                    status_lines=status,
                    smoketest=False,
                    scene_path=scene_path,
                    resume=resume,
                )
            )

    def _do_start_training(
        self,
        smoketest: bool = False,
        scene_path: str | None = None,
        resume: str | None = None,
    ):
        """Actually start training (push dashboard and kick off worker)."""
        dashboard = TrainingDashboard()
        self._dashboard = dashboard
        self._smoketest = smoketest
        self._scene_path = scene_path
        self._resume = resume
        # Route Python log records into the TUI log panel
        self._tui_log_handler = TuiLogHandler(self)
        logging.getLogger().addHandler(self._tui_log_handler)
        self.push_screen(dashboard)
        self._run_training()

    @work(thread=True, exclusive=True)
    def _run_training(self) -> None:
        from training.train import run_training

        # Force serial collection in TUI to avoid multiprocessing FD issues
        # (Textual's event loop holds file descriptors that become invalid
        # when multiprocessing.spawn tries to inherit them)
        num_workers = 1
        try:
            run_training(
                self,
                self.commands,
                smoketest=self._smoketest,
                num_workers=num_workers,
                scene_path=self._scene_path,
                resume=self._resume,
            )
        except KeyboardInterrupt:
            log.warning("Training interrupted by user")
            try:
                self.call_from_thread(
                    self.log_message,
                    "[bold yellow]Training interrupted by user[/bold yellow]",
                )
            except Exception:
                pass
        except Exception:
            log.exception("Training crashed in TUI worker thread")
            try:
                import traceback

                err = traceback.format_exc().splitlines()[-1]
                self.call_from_thread(
                    self.log_message,
                    f"[bold red]Training crashed![/bold red] {err}",
                )
            except Exception:
                pass
        finally:
            # Remove TUI log handler to avoid stale references
            if hasattr(self, "_tui_log_handler"):
                logging.getLogger().removeHandler(self._tui_log_handler)

    def send_command(self, cmd: str):
        self.commands.send(cmd)

    def update_metrics(self, batch: int, metrics: dict):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.update_metrics(batch, metrics)

    def log_message(self, text: str):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.log_message(text)

    def mark_finished(self):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.mark_finished()

    def ai_commentary(self, text: str):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.log_ai_commentary(text)

    def set_header(
        self,
        run_name: str,
        branch: str,
        algorithm: str,
        wandb_url: str | None,
        bot_name: str | None = None,
        experiment_hypothesis: str | None = None,
    ):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard.set_header(
                run_name,
                branch,
                algorithm,
                wandb_url,
                bot_name,
                experiment_hypothesis,
            )

    def set_total_batches(self, total: int | None):
        """Called from training thread via call_from_thread."""
        if self._dashboard:
            self._dashboard._total_batches = total


def _setup_logging() -> None:
    """Configure root logger level and install an excepthook.

    Per-run file logging is set up in train.py when the run directory is
    created.  This function only sets the root level and installs an
    excepthook so unhandled exceptions are captured by whatever handlers
    are active at the time.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Capture unhandled exceptions to the log
    _original_excepthook = sys.excepthook

    def _logging_excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            logging.critical(
                "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
            )
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _logging_excepthook


def _is_mjpython() -> bool:
    """Check if we're running under mjpython."""
    return "MJPYTHON_BIN" in os.environ


def _check_mjpython():
    """Warn if not launched via mjpython (needed for MuJoCo viewer on macOS)."""
    if shutil.which("mjpython") is None:
        return  # mjpython not installed, nothing to check
    if not _is_mjpython():
        print(
            "Warning: main.py should be launched with mjpython for viewer/play support.\n"
            "  Use: uv run mjpython main.py   (or: make)\n"
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MindSim - single entry point for all modes",
    )
    sub = parser.add_subparsers(dest="command")

    # scene
    sub.add_parser("scene", help="Scene gen preview (procedural furniture)")

    # view
    p_view = sub.add_parser("view", help="Launch MuJoCo viewer")
    p_view.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_view.add_argument(
        "--stage", type=int, default=None, help="Curriculum stage 1-4 (default: none)"
    )

    # play
    p_play = sub.add_parser("play", help="Play trained policy in viewer")
    p_play.add_argument(
        "checkpoint",
        nargs="?",
        default="latest",
        help="Checkpoint ref (default: latest)",
    )
    p_play.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_play.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run name to play (resolves checkpoint from run dir)",
    )

    # train
    p_train = sub.add_parser("train", help="Train (headless CLI, no TUI)")
    p_train.add_argument(
        "--smoketest", action="store_true", help="Fast end-to-end smoketest"
    )
    p_train.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_train.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    p_train.add_argument(
        "--num-workers", type=int, default=None, help="Number of parallel workers"
    )
    p_train.add_argument(
        "--no-dirty-check",
        action="store_true",
        help="Skip uncommitted-changes check",
    )

    # smoketest (alias for train --smoketest)
    sub.add_parser("smoketest", help="Alias for train --smoketest")

    # quicksim
    sub.add_parser("quicksim", help="Quick simulation with Rerun debug vis")

    # visualize
    p_viz = sub.add_parser("visualize", help="One-shot Rerun visualization")
    p_viz.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )
    p_viz.add_argument("--steps", type=int, default=1000, help="Number of sim steps")

    # validate-rewards
    p_vr = sub.add_parser(
        "validate-rewards", help="Validate reward hierarchy for a bot"
    )
    p_vr.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: all bots)"
    )

    # describe
    p_desc = sub.add_parser("describe", help="Print human-readable pipeline summary")
    p_desc.add_argument(
        "--bot", type=str, default=None, help="Bot name (default: simple2wheeler)"
    )

    # replay
    p_replay = sub.add_parser(
        "replay",
        help="Download or regenerate Rerun recordings for a run",
    )
    p_replay.add_argument("run", help="Run name (local or W&B)")
    p_replay.add_argument(
        "--batches",
        default=None,
        help="Comma-separated batch numbers (e.g. 500,1000,2000)",
    )
    p_replay.add_argument(
        "--last",
        type=int,
        default=None,
        help="Download only the N most recent recordings",
    )
    p_replay.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run episodes from checkpoints instead of downloading recordings",
    )
    p_replay.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for target placement (only with --regenerate)",
    )

    return parser


def _run_tui():
    """Launch the interactive Textual TUI."""
    app = MindSimApp()
    app.run()

    # Dispatch to selected mode after TUI exits.
    # Use os.execvp to replace the process entirely, giving GLFW a clean
    # AppKit state (Textual's event loop corrupts it on macOS).
    # Must go through "uv run mjpython" — sys.executable is the venv python,
    # not mjpython, because mjpython exec's the real interpreter after setup.
    if app.next_action == "view":
        cmd = ["uv", "run", "mjpython", "main.py", "view"]
        if app.next_scene:
            bot_name = Path(app.next_scene).parent.name
            cmd.extend(["--bot", bot_name])
        if app.next_stage:
            cmd.extend(["--stage", str(app.next_stage)])
        os.execvp("uv", cmd)

    elif app.next_action == "scene":
        cmd = ["uv", "run", "mjpython", "main.py", "scene"]
        os.execvp("uv", cmd)

    elif app.next_action == "play":
        cmd = ["uv", "run", "mjpython", "main.py", "play"]
        # If we have a specific checkpoint path (legacy), pass it directly
        if app.next_checkpoint_path:
            cmd.append(app.next_checkpoint_path)
        # If we have a run name, use --run
        elif app.next_run_name:
            cmd.extend(["--run", app.next_run_name])
        if app.next_scene:
            bot_name = Path(app.next_scene).parent.name
            cmd.extend(["--bot", bot_name])
        os.execvp("uv", cmd)


def main():
    _setup_logging()
    _check_mjpython()

    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        # No subcommand → interactive TUI
        _run_tui()

    elif args.command == "scene":
        from sim.scene_preview import run_scene_preview

        run_scene_preview()

    elif args.command == "view":
        scene_path = _resolve_scene_path(args.bot)
        from tools.view import run_view

        run_view(scene_path, stage=args.stage)

    elif args.command == "play":
        scene_path = _resolve_scene_path(args.bot)
        from tools.play import run_play

        # --run takes priority: resolve checkpoint from run directory
        checkpoint_ref = args.run if args.run else args.checkpoint
        run_play(checkpoint_ref=checkpoint_ref, scene_path=scene_path)

    elif args.command == "train":
        scene_path = _resolve_scene_path(args.bot)
        # Dirty-tree check (skip for smoketests and --no-dirty-check)
        if not args.smoketest and not args.no_dirty_check:
            clean, status = _git_is_clean()
            if not clean:
                print("Uncommitted changes:")
                print(status)
                print()
                while True:
                    choice = (
                        input("[c] Commit with Claude  [s] Start anyway  [q] Quit: ")
                        .strip()
                        .lower()
                    )
                    if choice == "q":
                        sys.exit(0)
                    elif choice == "s":
                        break
                    elif choice == "c":
                        if _run_claude_commit():
                            print("Worktree is clean. Starting training.")
                        else:
                            _, new_status = _git_is_clean()
                            print("Still dirty after commit attempt:")
                            print(new_status)
                            print()
                            continue
                        break
        from training.train import main as train_main

        train_main(
            smoketest=args.smoketest,
            bot=args.bot,
            resume=args.resume,
            num_workers=args.num_workers,
            scene_path=scene_path,
        )

    elif args.command == "smoketest":
        from training.train import main as train_main

        bots = _discover_bots()
        if not bots:
            print("Error: No bots found in bots/*/scene.xml", file=sys.stderr)
            sys.exit(1)
        for i, bot_info in enumerate(bots):
            name = bot_info["name"]
            print(f"\n{'=' * 60}")
            print(f"Smoketest [{i + 1}/{len(bots)}]: {name}")
            print(f"{'=' * 60}")
            train_main(smoketest=True, bot=name, scene_path=bot_info["scene_path"])
        print(f"\nAll {len(bots)} bots passed smoketest.")

    elif args.command == "quicksim":
        from tools.quick_sim import run_quick_sim

        run_quick_sim()

    elif args.command == "visualize":
        scene_path = _resolve_scene_path(args.bot)
        from viz.visualize import run_visualization

        run_visualization(scene_path=scene_path, num_steps=args.steps)

    elif args.command == "validate-rewards":
        _validate_rewards(args.bot)

    elif args.command == "describe":
        from training.pipeline import pipeline_for_bot

        bot_name = args.bot or "simple2wheeler"
        pipeline = pipeline_for_bot(bot_name)
        print(pipeline.describe())

    elif args.command == "replay":
        from viz.replay import run_replay

        batches = None
        if args.batches:
            batches = [int(b.strip()) for b in args.batches.split(",")]
        run_replay(
            args.run,
            batches=batches,
            seed=args.seed,
            regenerate=args.regenerate,
            last_n=args.last,
        )


def _validate_rewards(bot_name: str | None):
    """Print reward hierarchy summary and run dominance checks."""
    from training.pipeline import pipeline_for_bot
    from training.rewards import build_reward_hierarchy

    # If no bot specified, validate all bots
    if bot_name is None:
        bots = _discover_bots()
        bot_names = [b["name"] for b in bots]
    else:
        bot_names = [bot_name]

    for name in bot_names:
        cfg = pipeline_for_bot(name)
        hierarchy = build_reward_hierarchy(name, cfg.env)

        print(f"\nReward Hierarchy for {name}")
        print("=" * 50)
        print(hierarchy.summary_table())
        print()

        dom = hierarchy.dominance_check()
        if dom:
            print("Dominance check:")
            print(dom)
        else:
            print("Dominance check: (single priority level, no check needed)")
        print()

        # Scenario tests
        _run_scenario_tests(name, hierarchy, cfg)


def _run_scenario_tests(bot_name: str, hierarchy, cfg):
    """Run scenario tests showing per-step reward in different situations."""

    active = hierarchy.active_components()
    if not active:
        print("Scenario tests: no active components")
        return

    print("Scenario tests:")

    # Scenario 1: "Perfect stand" -- healthy, no movement
    stand_reward = 0.0
    for c in active:
        if c.name == "alive":
            stand_reward += c.scale * 1.0  # healthy
        elif c.name == "upright":
            stand_reward += c.scale * 1.0  # perfectly upright
        elif c.name == "time":
            stand_reward += c.scale * (-1.0)  # time penalty always applies
        # Everything else is 0 (no movement, no contact, etc.)
    print(f"  Perfect stand  (healthy, no movement):    {stand_reward:+.3f}/step")

    # Scenario 2: "Diving forward" -- unhealthy, fast forward
    dive_reward = 0.0
    for c in active:
        if c.name == "alive":
            dive_reward += 0.0  # unhealthy -- no alive bonus
        elif c.name == "forward_velocity":
            dive_reward += c.scale * 1.0  # max forward vel, but gated by up_z
            # If gated by is_healthy, diving doesn't earn this either
            if c.gated_by and "is_healthy" in c.gated_by:
                dive_reward -= c.scale * 1.0  # undo: not healthy
        elif c.name == "distance":
            dive_reward += c.scale * 0.5  # some distance progress
        elif c.name == "time":
            dive_reward += c.scale * (-1.0)
        elif c.name == "contact":
            dive_reward += c.scale * (-1.0)  # body on floor
    print(f"  Diving forward (unhealthy, fast):         {dive_reward:+.3f}/step")

    # Scenario 3: "Walking well" -- healthy, moderate forward, upright
    walk_reward = 0.0
    for c in active:
        if c.name == "alive":
            walk_reward += c.scale * 1.0
        elif c.name == "forward_velocity":
            walk_reward += c.scale * 0.5  # moderate speed
        elif c.name == "distance":
            walk_reward += c.scale * 0.3  # some progress
        elif c.name == "upright":
            walk_reward += c.scale * 0.9  # mostly upright
        elif c.name == "energy":
            walk_reward += c.scale * (-3.0)  # moderate energy
        elif c.name == "smoothness":
            walk_reward += c.scale * (-1.0)  # some jerk
        elif c.name == "time":
            walk_reward += c.scale * (-1.0)
    print(f"  Walking well   (healthy, forward):        {walk_reward:+.3f}/step")

    # Check key invariant: standing must beat diving
    if stand_reward > dive_reward:
        print(f"\n  Standing ({stand_reward:+.3f}) > Diving ({dive_reward:+.3f}) [ok]")
    else:
        print(
            f"\n  Standing ({stand_reward:+.3f}) <= Diving ({dive_reward:+.3f}) "
            "[!!] -- diving is more rewarding than standing!"
        )

    if walk_reward > stand_reward:
        print(
            f"  Walking ({walk_reward:+.3f}) > Standing ({stand_reward:+.3f}) "
            "[ok] -- walking is the best outcome"
        )
    else:
        print(
            f"  Walking ({walk_reward:+.3f}) <= Standing ({stand_reward:+.3f}) "
            "[!!] -- standing is better than walking"
        )


if __name__ == "__main__":
    main()
