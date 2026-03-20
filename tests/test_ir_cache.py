"""Tests for botcad.ir.cache.DiskCache."""

from __future__ import annotations

from pathlib import Path

from botcad.ir.cache import DiskCache
from botcad.ir.program import CadProgram


def _make_box_program(width: float = 0.06) -> CadProgram:
    """Helper: build a simple box program."""
    prog = CadProgram()
    box = prog.box(width, 0.04, 0.02)
    prog.query_volume(box)
    prog.output_ref = box
    return prog


class TestDiskCache:
    def test_miss_returns_none(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog = _make_box_program()
        assert cache.get(prog) is None

    def test_put_then_get(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog = _make_box_program()
        data = {"volume": 4.8e-5, "centroid": (0.0, 0.0, 0.0)}
        cache.put(prog, data)
        assert cache.get(prog) == data

    def test_different_programs_no_collision(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog_a = _make_box_program(width=0.06)
        prog_b = _make_box_program(width=0.10)

        cache.put(prog_a, {"volume": 1.0})
        cache.put(prog_b, {"volume": 2.0})

        assert cache.get(prog_a) == {"volume": 1.0}
        assert cache.get(prog_b) == {"volume": 2.0}

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        prog = _make_box_program()
        data = {"area": 0.01}

        DiskCache(cache_dir).put(prog, data)
        assert DiskCache(cache_dir).get(prog) == data

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog = _make_box_program()
        cache.put(prog, {"volume": 1.0})
        cache.invalidate(prog)
        assert cache.get(prog) is None

    def test_invalidate_missing_is_noop(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog = _make_box_program()
        cache.invalidate(prog)  # should not raise

    def test_clear(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path / "cache")
        prog_a = _make_box_program(width=0.06)
        prog_b = _make_box_program(width=0.10)

        cache.put(prog_a, {"volume": 1.0})
        cache.put(prog_b, {"volume": 2.0})
        cache.clear()

        assert cache.get(prog_a) is None
        assert cache.get(prog_b) is None
