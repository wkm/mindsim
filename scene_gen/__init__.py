"""Procedural scene generation for MuJoCo training environments.

Generates rooms with furniture from parametric concepts. Each concept
(table, chair, shelf, etc.) is a pure function: frozen params -> primitives.
Results are cached via @lru_cache for fast per-episode reuse.

Usage:
    from scene_gen import SceneComposer, PlacedObject, concepts

    composer = SceneComposer(model, data)
    scene = composer.random_scene()    # random furniture layout
    composer.apply(scene)              # write to MuJoCo model
    mujoco.mj_forward(model, data)     # update kinematics
"""

from scene_gen.composer import PlacedObject, SceneComposer, describe_scene, scene_id
from scene_gen.primitives import GeomType, Prim

__all__ = [
    "SceneComposer",
    "PlacedObject",
    "Prim",
    "GeomType",
    "describe_scene",
    "scene_id",
]
