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
    │                 │  Rewards (total, cumulative) │
    │  Camera View    ├──────────────────────────────┤
    │                 │  Distance to target          │
    ├─────────────────┼──────────────────────────────┤
    │                 │  Actions (left, right motor) │
    │    3D Scene     │                              │
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
                    name="Rewards",
                    origin="/",
                    contents=[
                        "eval/reward/total",
                        "eval/reward/cumulative",
                    ],
                ),
                rrb.TimeSeriesView(
                    name="Distance",
                    origin="/",
                    contents=[
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
                row_shares=[1, 1, 1],
            ),
            column_shares=[1, 2],
        ),
        collapse_panels=False,  # Keep sources panel open for episode navigation
    )
