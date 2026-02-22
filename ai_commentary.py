"""
AI commentary for training runs.

Uses the Claude CLI in print mode with session continuity to provide
periodic analysis of training metrics — trend spotting, entropy collapse
warnings, plateau detection, etc.

Each training run gets a unique session ID so Claude can reference
earlier observations when generating new commentary.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    """Identity and context for a training run."""

    run_name: str
    bot_name: str
    algorithm: str
    policy_type: str
    hypothesis: str | None
    wandb_url: str | None
    git_branch: str
    config_summary: str


class AICommentator:
    """Generates periodic AI commentary on training progress.

    Uses `claude -p` with `--session-id` (first call) then `--resume`
    (subsequent calls) for conversation continuity within a run.
    """

    def __init__(self, config, run_context: RunContext):
        self._config = config
        self._context = run_context
        self._session_id = str(uuid.uuid4())
        self._call_count = 0
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        ctx = self._context
        lines = [
            "You are a training coach watching a reinforcement learning run in real time.",
            "Your job: observe metrics snapshots, spot trends, flag concerns, and suggest what to watch.",
            "",
            f"Run: {ctx.run_name}",
            f"Bot: {ctx.bot_name}",
            f"Algorithm: {ctx.algorithm}",
            f"Policy: {ctx.policy_type}",
            f"Branch: {ctx.git_branch}",
        ]
        if ctx.hypothesis:
            lines.append(f"Experiment hypothesis: {ctx.hypothesis}")
        if ctx.wandb_url:
            lines.append(f"W&B dashboard: {ctx.wandb_url}")
            lines.append(
                "You can use WebFetch to browse the W&B URL for charts and detailed metrics."
            )
        lines.append("")
        lines.append(f"Config:\n{ctx.config_summary}")
        lines.append("")
        lines.append("Instructions:")
        lines.append("- Be concise: 2-4 sentences per commentary.")
        lines.append(
            "- Note trends across snapshots (you'll see multiple over the run)."
        )
        lines.append(
            "- Flag concerns: entropy collapse, reward plateaus, KL spikes, etc."
        )
        lines.append("- Suggest what to watch or consider adjusting.")
        lines.append("- Use plain text, no markdown headers. Keep it conversational.")
        return "\n".join(lines)

    def generate_commentary(
        self, batch: int, metrics: dict, elapsed_secs: float
    ) -> str | None:
        """Generate commentary for the current training state.

        Returns the commentary text, or None on failure.
        """
        prompt = self._format_metrics_snapshot(batch, metrics, elapsed_secs)

        cmd = [
            "claude",
            "-p",
            "--model",
            self._config.model,
            "--allowedTools",
            "WebFetch Read Grep",
        ]

        if self._call_count == 0:
            cmd.extend(["--session-id", self._session_id])
            cmd.extend(["--system-prompt", self._system_prompt])
        else:
            cmd.extend(["--resume", self._session_id])

        cmd.append(prompt)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            self._call_count += 1
            text = result.stdout.strip()
            if text:
                return text
            log.warning("AI commentary returned empty output")
            return None
        except subprocess.TimeoutExpired:
            log.warning("AI commentary timed out (60s)")
            return None
        except subprocess.CalledProcessError as e:
            log.warning(
                "AI commentary failed: %s", e.stderr[:200] if e.stderr else str(e)
            )
            return None
        except FileNotFoundError:
            log.warning("claude CLI not found — AI commentary disabled")
            return None

    def _format_metrics_snapshot(
        self, batch: int, metrics: dict, elapsed_secs: float
    ) -> str:
        """Format current metrics as a human-readable prompt."""
        m = metrics
        elapsed_min = elapsed_secs / 60

        lines = [f"=== Batch {batch} ({elapsed_min:.1f} min elapsed) ==="]

        # Episode performance
        lines.append(f"Avg reward: {m.get('avg_reward', '?'):.2f}")
        lines.append(
            f"Best/worst reward: {m.get('best_reward', '?'):.2f} / {m.get('worst_reward', '?'):.2f}"
        )
        lines.append(f"Avg distance to target: {m.get('avg_distance', '?'):.2f}m")
        lines.append(f"Avg episode steps: {m.get('avg_steps', '?'):.0f}")

        # Success rates
        lines.append(
            f"Eval success (rolling): {m.get('rolling_eval_success_rate', 0):.0%}"
        )
        lines.append(f"Eval success (batch): {m.get('eval_success_rate', 0):.0%}")
        lines.append(f"Train success (batch): {m.get('batch_success_rate', 0):.0%}")

        # Optimization
        lines.append(f"Entropy: {m.get('entropy', '?')}")
        lines.append(f"Grad norm: {m.get('grad_norm', '?')}")
        if "policy_loss" in m:
            lines.append(f"Policy loss: {m['policy_loss']:.4f}")
        if "value_loss" in m:
            lines.append(f"Value loss: {m['value_loss']:.4f}")
        if "clip_fraction" in m:
            lines.append(f"Clip fraction: {m['clip_fraction']:.3f}")
        if "approx_kl" in m:
            lines.append(f"Approx KL: {m['approx_kl']:.4f}")

        # Policy std
        ps = m.get("policy_std")
        if ps is not None:
            try:
                lines.append(f"Policy std: {' / '.join(f'{s:.3f}' for s in ps)}")
            except TypeError:
                lines.append(f"Policy std: {ps:.3f}")

        # Curriculum
        lines.append(
            f"Curriculum stage: {m.get('curriculum_stage', '?')} / {m.get('num_stages', '?')}"
        )
        lines.append(f"Stage progress: {m.get('stage_progress', 0):.2f}")
        lines.append(
            f"Mastery: {m.get('mastery_count', 0)} / {m.get('mastery_batches', '?')}"
        )

        # Timing
        lines.append(f"Batch time: {m.get('batch_time', 0):.1f}s")
        lines.append(f"Episodes so far: {m.get('episode_count', '?')}")

        return "\n".join(lines)
