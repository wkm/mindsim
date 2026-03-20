"""Tag registry for tracking named edges through IR operations.

Tags are declared on primitive ops (e.g. a cylinder tagged "pocket").
When that shape participates in boolean ops, the resulting shape inherits
the tag. The OCCT backend uses tags to resolve edge selections for
fillet/chamfer operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from botcad.ir.ops import ShapeRef


@dataclass
class TagRegistry:
    """Tracks tag declarations and propagation through operations."""

    # tag_name -> source ShapeRef (the primitive that declared it)
    _declarations: dict[str, ShapeRef] = field(default_factory=dict)
    # ShapeRef -> set of tag names present on that shape
    _shape_tags: dict[ShapeRef, set[str]] = field(default_factory=dict)

    def declare(self, tag: str, ref: ShapeRef) -> None:
        """Declare a new tag on a primitive shape."""
        self._declarations[tag] = ref
        self._shape_tags.setdefault(ref, set()).add(tag)

    def source_ref(self, tag: str) -> ShapeRef:
        """Get the original ShapeRef that declared this tag."""
        if tag not in self._declarations:
            raise KeyError(f"Unknown tag: {tag!r}")
        return self._declarations[tag]

    def tags_on(self, ref: ShapeRef) -> frozenset[str]:
        """Get all tags present on a shape (including propagated)."""
        return frozenset(self._shape_tags.get(ref, set()))

    def propagate_boolean(
        self,
        result_ref: ShapeRef,
        target_ref: ShapeRef,
        tool_ref: ShapeRef,
    ) -> None:
        """Result of boolean op inherits tags from both operands."""
        inherited = set()
        inherited.update(self._shape_tags.get(target_ref, set()))
        inherited.update(self._shape_tags.get(tool_ref, set()))
        if inherited:
            self._shape_tags.setdefault(result_ref, set()).update(inherited)

    def propagate_transform(self, result_ref: ShapeRef, source_ref: ShapeRef) -> None:
        """Result of transform inherits tags from source."""
        source_tags = self._shape_tags.get(source_ref, set())
        if source_tags:
            self._shape_tags.setdefault(result_ref, set()).update(source_tags)
