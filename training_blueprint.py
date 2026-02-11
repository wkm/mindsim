"""
Rerun blueprint for eval episode visualization.

Imported by rerun_wandb.py to embed blueprint into recordings.
Shows deterministic policy behavior (no exploration noise).
"""

import rerun.blueprint as rrb


def create_training_blueprint():
    """
    Create a blueprint that organizes eval episode visualization.

    Uses deterministic eval episodes (mean actions, no sampling)
    to show true policy capability without exploration noise.

    Layout:
    ┌─────────────────┬──────────────────────────────┐
    │                 │  Value Function (V, returns) │
    │  Camera View    ├──────────────────────────────┤
    │                 │  Advantage                   │
    ├─────────────────┼──────────────────────────────┤
    │                 │  Rewards + Distance           │
    │    3D Scene     ├──────────────────────────────┤
    │                 │  Actions (left, right motor) │
    └─────────────────┴──────────────────────────────┘
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left column: visual views
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Camera",
                    origin="eval/camera",
                ),
                rrb.Spatial3DView(
                    name="Scene",
                    origin="eval",
                ),
                row_shares=[1, 1],
            ),
            # Right column: time series metrics
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Value Function",
                    origin="/",
                    contents=[
                        "eval/value/V_s",
                        "eval/value/cumulative_reward",
                        "eval/value/gae_return",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Advantage",
                    origin="/",
                    contents=[
                        "eval/value/advantage",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Rewards & Distance",
                    origin="/",
                    contents=[
                        "eval/reward/total",
                        "eval/reward/cumulative",
                        "eval/distance_to_target",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Actions",
                    origin="/",
                    contents=[
                        "eval/action/left_motor",
                        "eval/action/right_motor",
                    ],
                ),
                row_shares=[2, 1, 1, 1],
            ),
            column_shares=[1, 2],
        ),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Hidden),
    )
