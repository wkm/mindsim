"""Disk cache for ShapeScript execution results.

Keyed by ShapeScript.content_hash() (SHA-256). Stores arbitrary picklable
data (dicts with volume, centroid, area, inertia, stl_bytes, etc.) as
individual .pkl files in a cache directory.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from botcad.shapescript.program import ShapeScript

DEFAULT_CACHE_DIR = Path(".botcad_cache")


class DiskCache:
    """Simple file-backed cache for ShapeScript execution results.

    Each entry is stored as ``<content_hash>.pkl`` inside *cache_dir*.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, prog: ShapeScript) -> Path:
        return self.cache_dir / f"{prog.content_hash()}.pkl"

    def get(self, prog: ShapeScript) -> Any | None:
        """Return cached data for *prog*, or ``None`` on miss."""
        path = self._path_for(prog)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)  # noqa: S301

    def put(self, prog: ShapeScript, data: Any) -> None:
        """Store *data* (must be picklable) for *prog*."""
        path = self._path_for(prog)
        with path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def invalidate(self, prog: ShapeScript) -> None:
        """Remove the cached entry for *prog*, if any."""
        path = self._path_for(prog)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Remove all cached entries."""
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
