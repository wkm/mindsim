"""Fast tests for the scene generation system.

Validates that:
- Every registered concept generates valid primitives
- Every concept has a VARIATIONS dict
- The composer can place objects without errors
- Every concept has a rendered catalog PNG in docs/concepts/
- Room archetypes produce valid scenes
"""

from pathlib import Path

import mujoco
import pytest

from scene_gen import SceneComposer, concepts
from scene_gen.archetypes import ARCHETYPES, list_archetypes
from scene_gen.composer import describe_scene
from scene_gen.groupings import GROUPINGS
from scene_gen.primitives import GeomType, Prim, footprint, obb_overlaps
from scene_gen.room import WALL_THICKNESS, InteriorWall, RoomConfig, random_room_config

DOCS_CONCEPTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "concepts"


# ---------------------------------------------------------------------------
# Concept validation
# ---------------------------------------------------------------------------


class TestConcepts:
    """Every concept module must generate valid prims."""

    @pytest.fixture(params=concepts.list_concepts())
    def concept_name(self, request):
        return request.param

    def test_has_params_and_generate(self, concept_name):
        mod = concepts.get(concept_name)
        assert hasattr(mod, "Params"), f"{concept_name} missing Params dataclass"
        assert hasattr(mod, "generate"), f"{concept_name} missing generate()"

    def test_has_variations(self, concept_name):
        mod = concepts.get(concept_name)
        variations = getattr(mod, "VARIATIONS", None)
        assert variations is not None, f"{concept_name} missing VARIATIONS dict"
        assert len(variations) >= 1, f"{concept_name} VARIATIONS is empty"

    def test_default_generates_valid_prims(self, concept_name):
        mod = concepts.get(concept_name)
        prims = mod.generate(mod.Params())
        assert isinstance(prims, tuple), "generate() must return a tuple"
        assert len(prims) >= 1, "generate() must return at least 1 prim"
        assert len(prims) <= 8, f"generate() returned {len(prims)} prims (max 8)"
        for p in prims:
            assert isinstance(p, Prim)
            assert isinstance(p.geom_type, GeomType)
            assert len(p.size) == 3
            assert len(p.pos) == 3
            assert len(p.rgba) == 4
            assert all(0 <= c <= 1 for c in p.rgba), f"rgba out of range: {p.rgba}"

    def test_all_variations_generate(self, concept_name):
        mod = concepts.get(concept_name)
        variations = getattr(mod, "VARIATIONS", {})
        for var_name, params in variations.items():
            prims = mod.generate(params)
            assert isinstance(prims, tuple), f"{concept_name}/{var_name}: not a tuple"
            assert 1 <= len(prims) <= 8, (
                f"{concept_name}/{var_name}: {len(prims)} prims"
            )

    def test_footprint_positive(self, concept_name):
        mod = concepts.get(concept_name)
        prims = mod.generate(mod.Params())
        hx, hy = footprint(prims)
        assert hx > 0, f"{concept_name}: footprint hx={hx}"
        assert hy > 0, f"{concept_name}: footprint hy={hy}"

    def test_has_placement(self, concept_name):
        mod = concepts.get(concept_name)
        assert hasattr(mod, "PLACEMENT"), f"{concept_name} missing PLACEMENT"


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


class TestCatalogCompleteness:
    """Every concept must have a rendered PNG in docs/concepts/."""

    def test_catalog_png_exists(self):
        missing = []
        for name in concepts.list_concepts():
            png = DOCS_CONCEPTS_DIR / f"{name}.png"
            if not png.exists():
                missing.append(name)
        assert not missing, (
            f"Missing catalog PNGs for: {', '.join(missing)}. "
            f"Run: uv run python -m scene_gen.render_catalog"
        )


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


class TestComposer:
    """SceneComposer placement and apply."""

    @pytest.fixture(scope="class")
    def composer(self):
        spec = mujoco.MjSpec.from_file("worlds/room.xml")
        SceneComposer.prepare_spec(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return SceneComposer(model, data)

    def test_slot_count(self, composer):
        assert composer.max_objects >= 16
        assert composer.max_geoms_per_object == 8

    def test_random_scene_pure(self, composer):
        scene = composer.random_scene(seed=42)
        assert len(scene) >= 1
        assert all(hasattr(o, "concept") for o in scene)

    def test_random_scene_deterministic(self, composer):
        s1 = composer.random_scene(seed=123)
        s2 = composer.random_scene(seed=123)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.concept == b.concept
            assert a.pos == b.pos

    def test_apply_and_clear(self, composer):
        scene = composer.random_scene(seed=42)
        composer.apply(scene)
        composer.clear()

    def test_geom_positions_survive_mj_step(self, composer):
        """Geom world positions must be correct after mj_step.

        Regression test: MuJoCo skips the geom-level transform for geoms
        compiled with pos=[0,0,0] (a "same frame" optimization). The old
        one-body-per-object layout hit this bug — geoms appeared flat on
        the floor in the viewer because mj_step overwrote patched positions.
        The one-body-per-geom layout avoids this by using body_pos for
        world positioning and keeping geom_pos at [0,0,0] (compile-time).
        """
        import numpy as np

        scene = composer.random_scene(seed=42)
        composer.apply(scene)

        # Record positions after apply (which calls mj_forward internally)
        model, data = composer.model, composer.data
        pre_xpos = data.geom_xpos.copy()

        # Simulate what the viewer does: mj_step overwrites data.geom_xpos
        mujoco.mj_step(model, data)

        # Every visible geom must still be at the same position
        for obj in scene:
            concept_mod = concepts.get(obj.concept)
            prims = concept_mod.generate(obj.params)
            for prim in prims:
                if any(c > 0 for c in prim.rgba):  # visible
                    # At least one geom should be at nonzero z (furniture has height)
                    pass

        # The key assertion: positions must match after mj_step
        assert np.allclose(data.geom_xpos, pre_xpos, atol=1e-6), (
            "geom_xpos changed after mj_step — "
            "mj_kinematics is not preserving runtime geom positions"
        )

    def test_describe_scene(self, composer):
        scene = composer.random_scene(seed=42)
        desc = describe_scene(scene, seed=42)
        assert "Scene #" in desc
        assert "objects" in desc


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------


class TestArchetypes:
    """Room archetypes produce valid concept lists."""

    @pytest.fixture(scope="class")
    def composer(self):
        spec = mujoco.MjSpec.from_file("worlds/room.xml")
        SceneComposer.prepare_spec(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return SceneComposer(model, data)

    @pytest.fixture(params=list_archetypes())
    def archetype_name(self, request):
        return request.param

    def test_archetype_generates_scene(self, composer, archetype_name):
        scene = composer.random_scene(archetype=archetype_name, seed=42)
        assert len(scene) >= 1

    def test_archetype_required_concepts_present(self, archetype_name):
        """Required concepts should always appear in the archetype's sample."""
        import numpy as np

        arch = ARCHETYPES[archetype_name]
        rng = np.random.default_rng(42)
        picks = arch.sample(rng, max_objects=16)
        for slot in arch.required:
            count = sum(1 for p in picks if p == slot.concept)
            assert count >= slot.count, (
                f"{archetype_name}: required {slot.concept} x{slot.count}, got {count}"
            )

    def test_random_archetype(self, composer):
        scene = composer.random_scene(archetype="random", seed=42)
        assert len(scene) >= 1


# ---------------------------------------------------------------------------
# Groupings
# ---------------------------------------------------------------------------


class TestGroupings:
    """Grouping definitions reference valid concepts."""

    def test_grouping_anchors_are_valid_concepts(self):
        available = set(concepts.list_concepts())
        for gname, grouping in GROUPINGS.items():
            assert grouping.anchor in available, (
                f"Grouping '{gname}' anchor '{grouping.anchor}' not a concept"
            )

    def test_grouping_satellites_are_valid_concepts(self):
        available = set(concepts.list_concepts())
        for gname, grouping in GROUPINGS.items():
            for sat in grouping.satellites:
                assert sat.concept in available, (
                    f"Grouping '{gname}' satellite '{sat.concept}' not a concept"
                )


# ---------------------------------------------------------------------------
# Room layout (walls + variable size)
# ---------------------------------------------------------------------------


class TestRoom:
    """Variable room size and interior walls."""

    @pytest.fixture(scope="class")
    def composer(self):
        spec = mujoco.MjSpec.from_file("worlds/room.xml")
        SceneComposer.prepare_spec(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return SceneComposer(model, data)

    def test_wall_slots_discovered(self, composer):
        """Room wall slots should be found in the compiled model."""
        assert composer._room.has_slots

    def test_random_room_config(self):
        """random_room_config should produce valid configs."""
        import numpy as np

        rng = np.random.default_rng(42)
        config = random_room_config(rng)
        assert 2.5 <= config.half_extent <= 4.0
        for wall in config.interior_walls:
            assert wall.axis in ("x", "y")
            assert wall.door_width > 0

    def test_scene_with_room_config(self, composer):
        """Scenes with explicit room config should work."""
        config = RoomConfig(half_extent=3.0)
        scene = composer.random_scene(seed=42, room_config=config)
        composer.apply(scene)
        assert len(scene) >= 1

    def test_scene_with_interior_wall(self, composer):
        """Scenes with an interior wall + doorway should work."""
        config = RoomConfig(
            half_extent=3.5,
            interior_walls=(
                InteriorWall(axis="x", offset=0.5, door_pos=0.0, door_width=1.2),
            ),
        )
        scene = composer.random_scene(seed=42, room_config=config)
        composer.apply(scene)
        assert len(scene) >= 1

    def test_variable_room_size_auto(self, composer):
        """When no room_config is passed, random_scene auto-generates one."""
        scene = composer.random_scene(seed=99)
        # The last room config should have been stored
        assert composer._last_room_config is not None
        assert composer._last_room_config.half_extent > 0
        composer.apply(scene)

    def test_small_room_fewer_objects(self, composer):
        """A small room should still produce a valid scene."""
        config = RoomConfig(half_extent=2.5)
        scene = composer.random_scene(seed=42, room_config=config)
        composer.apply(scene)
        assert len(scene) >= 1

    def test_walls_survive_mj_step(self, composer):
        """Wall positions must persist after mj_step (same as furniture)."""
        import numpy as np

        config = RoomConfig(half_extent=3.0)
        scene = composer.random_scene(seed=42, room_config=config)
        composer.apply(scene)

        model, data = composer.model, composer.data
        pre_xpos = data.geom_xpos.copy()
        mujoco.mj_step(model, data)
        assert np.allclose(data.geom_xpos, pre_xpos, atol=1e-6)

    def test_objects_inside_exterior_walls(self, composer):
        """Every placed object's rotated bbox must be fully inside walls.

        Generates many scenes with various room sizes and verifies that no
        object's bounding box extends past the exterior walls.
        """
        import numpy as np

        room_sizes = [2.5, 3.0, 3.5, 4.0]
        seeds = range(50)

        for he in room_sizes:
            config = RoomConfig(half_extent=he)
            limit = he - WALL_THICKNESS
            for seed in seeds:
                scene = composer.random_scene(seed=seed, room_config=config)
                for obj in scene:
                    concept_mod = concepts.get(obj.concept)
                    prims = concept_mod.generate(obj.params)
                    hx, hy = footprint(prims)
                    cx, cy = obj.pos
                    rot = obj.rotation

                    # Compute rotated extent along each axis
                    cos_r = abs(np.cos(rot))
                    sin_r = abs(np.sin(rot))
                    reach_x = hx * cos_r + hy * sin_r
                    reach_y = hx * sin_r + hy * cos_r

                    assert abs(cx) + reach_x < limit, (
                        f"Object {obj.concept} at ({cx:.2f}, {cy:.2f}) "
                        f"rot={np.degrees(rot):.0f} extends past X wall "
                        f"(reach={reach_x:.2f}, limit={limit:.2f}, "
                        f"he={he}, seed={seed})"
                    )
                    assert abs(cy) + reach_y < limit, (
                        f"Object {obj.concept} at ({cx:.2f}, {cy:.2f}) "
                        f"rot={np.degrees(rot):.0f} extends past Y wall "
                        f"(reach={reach_y:.2f}, limit={limit:.2f}, "
                        f"he={he}, seed={seed})"
                    )

    def test_objects_clear_of_interior_walls(self, composer):
        """No placed object should overlap with interior wall segments.

        Objects CAN be placed in the doorway area — only the solid wall
        segments should block placement.
        """

        # Test with both axis orientations and various configs
        configs = [
            RoomConfig(
                half_extent=3.5,
                interior_walls=(
                    InteriorWall(axis="x", offset=0.5, door_pos=0.0, door_width=1.2),
                ),
            ),
            RoomConfig(
                half_extent=3.0,
                interior_walls=(
                    InteriorWall(axis="y", offset=-0.3, door_pos=0.5, door_width=1.0),
                ),
            ),
            RoomConfig(
                half_extent=2.5,
                interior_walls=(
                    InteriorWall(axis="x", offset=0.0, door_pos=-0.5, door_width=0.8),
                ),
            ),
        ]

        for config in configs:
            he = config.half_extent
            # Build wall segment OBBs (same logic as composer)
            wall_obbs = []
            for iwall in config.interior_walls:
                door_half = iwall.door_width / 2
                if iwall.axis == "x":
                    seg1_s, seg1_e = -he, iwall.door_pos - door_half
                    seg2_s, seg2_e = iwall.door_pos + door_half, he
                    if seg1_e > seg1_s + 0.05:
                        seg1_hx = (seg1_e - seg1_s) / 2
                        seg1_cx = (seg1_s + seg1_e) / 2
                        wall_obbs.append(
                            (seg1_cx, iwall.offset, seg1_hx, WALL_THICKNESS, 0.0)
                        )
                    if seg2_e > seg2_s + 0.05:
                        seg2_hx = (seg2_e - seg2_s) / 2
                        seg2_cx = (seg2_s + seg2_e) / 2
                        wall_obbs.append(
                            (seg2_cx, iwall.offset, seg2_hx, WALL_THICKNESS, 0.0)
                        )
                else:  # axis == "y"
                    seg1_s, seg1_e = -he, iwall.door_pos - door_half
                    seg2_s, seg2_e = iwall.door_pos + door_half, he
                    if seg1_e > seg1_s + 0.05:
                        seg1_hy = (seg1_e - seg1_s) / 2
                        seg1_cy = (seg1_s + seg1_e) / 2
                        wall_obbs.append(
                            (iwall.offset, seg1_cy, WALL_THICKNESS, seg1_hy, 0.0)
                        )
                    if seg2_e > seg2_s + 0.05:
                        seg2_hy = (seg2_e - seg2_s) / 2
                        seg2_cy = (seg2_s + seg2_e) / 2
                        wall_obbs.append(
                            (iwall.offset, seg2_cy, WALL_THICKNESS, seg2_hy, 0.0)
                        )

            for seed in range(50):
                scene = composer.random_scene(seed=seed, room_config=config)
                for obj in scene:
                    concept_mod = concepts.get(obj.concept)
                    prims = concept_mod.generate(obj.params)
                    hx, hy = footprint(prims)
                    cx, cy = obj.pos
                    rot = obj.rotation

                    for wcx, wcy, whx, why, wrot in wall_obbs:
                        assert not obb_overlaps(
                            cx,
                            cy,
                            hx,
                            hy,
                            rot,
                            wcx,
                            wcy,
                            whx,
                            why,
                            wrot,
                            margin=0.0,
                        ), (
                            f"Object {obj.concept} at ({cx:.2f}, {cy:.2f}) "
                            f"overlaps interior wall segment at "
                            f"({wcx:.2f}, {wcy:.2f}), "
                            f"he={he}, seed={seed}"
                        )

    def test_small_room_wall_clearance(self, composer):
        """Stress test: small room with interior wall still places objects."""
        config = RoomConfig(
            half_extent=2.5,
            interior_walls=(
                InteriorWall(axis="x", offset=0.0, door_pos=0.0, door_width=1.0),
            ),
        )
        # Should still produce at least 1 object across multiple seeds
        placed_any = False
        for seed in range(20):
            scene = composer.random_scene(seed=seed, room_config=config)
            if len(scene) >= 1:
                placed_any = True
                break
        assert placed_any, "Small room with interior wall couldn't place any objects"
