"""
Rerun blueprint for training visualization.

Imported by rerun_wandb.py to embed blueprint into recordings.
"""

import rerun.blueprint as rrb


def create_training_blueprint():
    """
    Create a blueprint that organizes training metrics into logical groups.

    Layout:
    ┌─────────────────┬──────────────────────────────┐
    │                 │  Rewards (total, cumulative) │
    │  Camera View    ├──────────────────────────────┤
    │                 │  Distance (to target, moved) │
    ├─────────────────┼──────────────────────────────┤
    │                 │  Actions (left, right motor) │
    │    3D Scene     ├──────────────────────────────┤
    │                 │  Policy (std, log_prob)      │
    └─────────────────┴──────────────────────────────┘
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            # Left column: visual views
            rrb.Vertical(
                rrb.Spatial2DView(
                    name="Camera",
                    origin="training/camera",
                ),
                rrb.Spatial3DView(
                    name="Scene",
                    origin="training",
                ),
                row_shares=[1, 1],
            ),
            # Right column: time series metrics
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Rewards",
                    origin="/",
                    contents=[
                        "training/reward/total",
                        "training/reward/cumulative",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Distance",
                    origin="/",
                    contents=[
                        "training/distance_to_target",
                        "training/distance_moved",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Actions",
                    origin="/",
                    contents=[
                        "training/action/left_motor",
                        "training/action/right_motor",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Policy",
                    origin="/",
                    contents=[
                        "training/policy/std_left",
                        "training/policy/std_right",
                        "training/policy/log_prob",
                    ],
                ),
                row_shares=[1, 1, 1, 1],
            ),
            column_shares=[1, 2],
        ),
        collapse_panels=False,  # Keep sources panel open for episode navigation
    )
