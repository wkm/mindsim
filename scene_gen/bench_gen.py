"""Benchmark scene generation speed.

Usage:
    uv run python -m scene_gen.bench_gen              # 1000 scenes, default settings
    uv run python -m scene_gen.bench_gen --count 5000  # more scenes
    uv run python -m scene_gen.bench_gen --objects 8    # force 8 objects per scene
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from scene_gen import SceneComposer

ROOM_XML = str(Path(__file__).resolve().parent.parent / "worlds" / "room.xml")


def bench(n_scenes: int, n_objects: int | None, apply: bool) -> dict:
    """Run the benchmark. Returns timing stats."""
    spec = mujoco.MjSpec.from_file(ROOM_XML)
    SceneComposer.prepare_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    composer = SceneComposer(model, data)

    rng = np.random.default_rng(42)
    seeds = [int(rng.integers(0, 2**32)) for _ in range(n_scenes)]

    gen_times: list[float] = []
    apply_times: list[float] = []
    object_counts: list[int] = []

    for seed in seeds:
        t0 = time.perf_counter()
        scene = composer.random_scene(seed=seed, n_objects=n_objects)
        t1 = time.perf_counter()
        gen_times.append(t1 - t0)
        object_counts.append(len(scene))

        if apply:
            t2 = time.perf_counter()
            composer.apply(scene)
            t3 = time.perf_counter()
            apply_times.append(t3 - t2)

    gen_arr = np.array(gen_times) * 1000  # ms
    obj_arr = np.array(object_counts)

    stats = {
        "n_scenes": n_scenes,
        "gen_mean_ms": float(np.mean(gen_arr)),
        "gen_median_ms": float(np.median(gen_arr)),
        "gen_p95_ms": float(np.percentile(gen_arr, 95)),
        "gen_p99_ms": float(np.percentile(gen_arr, 99)),
        "gen_total_s": float(np.sum(gen_arr) / 1000),
        "objects_mean": float(np.mean(obj_arr)),
        "objects_max": int(np.max(obj_arr)),
        "scenes_per_sec": n_scenes / (np.sum(gen_arr) / 1000),
    }

    if apply_times:
        app_arr = np.array(apply_times) * 1000
        stats["apply_mean_ms"] = float(np.mean(app_arr))
        stats["apply_median_ms"] = float(np.median(app_arr))
        stats["total_mean_ms"] = stats["gen_mean_ms"] + stats["apply_mean_ms"]

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark scene generation")
    parser.add_argument("--count", type=int, default=1000, help="Number of scenes")
    parser.add_argument(
        "--objects", type=int, default=None, help="Force N objects per scene"
    )
    parser.add_argument(
        "--no-apply", action="store_true", help="Skip apply+mj_forward (gen only)"
    )
    args = parser.parse_args()

    print(f"Benchmarking {args.count} scenes...")
    stats = bench(args.count, args.objects, apply=not args.no_apply)

    print(f"\n  scenes:          {stats['n_scenes']}")
    print(
        f"  objects/scene:   {stats['objects_mean']:.1f} avg, {stats['objects_max']} max"
    )
    print(f"  gen mean:        {stats['gen_mean_ms']:.3f} ms")
    print(f"  gen median:      {stats['gen_median_ms']:.3f} ms")
    print(f"  gen p95:         {stats['gen_p95_ms']:.3f} ms")
    print(f"  gen p99:         {stats['gen_p99_ms']:.3f} ms")
    print(f"  gen total:       {stats['gen_total_s']:.2f} s")
    print(f"  scenes/sec:      {stats['scenes_per_sec']:.0f}")

    if "apply_mean_ms" in stats:
        print(f"  apply mean:      {stats['apply_mean_ms']:.3f} ms")
        print(
            f"  total mean:      {stats['total_mean_ms']:.3f} ms  (gen + apply + fwd)"
        )


if __name__ == "__main__":
    main()
