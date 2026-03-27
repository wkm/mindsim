"""Project-specific linting rules enforced via Python AST analysis.

Run as part of `make lint`. Each rule encodes a lesson learned from
bugs that static analysis (ruff/mypy) cannot catch — project conventions
about orientation, placement, and data flow.

Add new rules as functions decorated with @rule. Each rule receives an
ast.AST node and a Context, and yields Violation tuples.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

BOTCAD_ROOT = Path(__file__).parent.parent / "botcad"
RULES: list[Rule] = []


@dataclass
class Violation:
    path: Path
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


RuleCheck = type[ast.NodeVisitor]


@dataclass
class Rule:
    name: str
    check: RuleCheck


def rule(name: str):
    """Register an AST NodeVisitor class as a lint rule."""

    def decorator(cls: type[ast.NodeVisitor]) -> type[ast.NodeVisitor]:
        RULES.append(Rule(name=name, check=cls))
        return cls

    return decorator


def _run_rules(path: Path, tree: ast.AST) -> Iterator[Violation]:
    """Run all registered rules against a parsed AST."""
    for r in RULES:
        visitor = r.check()
        visitor._violations: list[Violation] = []  # type: ignore[attr-defined]
        visitor._path = path  # type: ignore[attr-defined]
        visitor._rule_name = r.name  # type: ignore[attr-defined]
        visitor.visit(tree)
        yield from visitor._violations  # type: ignore[attr-defined]


class LintVisitor(ast.NodeVisitor):
    """Base class for lint rules. Provides helpers for emitting violations."""

    _violations: list[Violation]
    _path: Path
    _rule_name: str

    def _warn(self, node: ast.AST, message: str) -> None:
        self._violations.append(
            Violation(
                path=self._path,
                line=getattr(node, "lineno", 0),
                rule=self._rule_name,
                message=message,
            )
        )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@rule("no-rotation-between-in-emitters")
class NoRotationBetweenInEmitters(LintVisitor):
    """Ban direct rotation_between() calls in emitter modules.

    Lesson: viewer.py and mujoco.py independently computed fastener
    orientations using rotation_between() with different axis conventions,
    causing Pi/camera/wheel fasteners to face wrong directions.

    Fastener orientation should go through a single function (fastener_pose)
    in geometry.py, not be re-derived per-emitter.
    """

    # Only applies to emitter modules
    EMITTER_PATHS = {"emit/viewer.py", "emit/mujoco.py", "emit/cad.py"}

    def visit_Call(self, node: ast.Call) -> None:
        # Check if this file is an emitter
        rel = str(self._path.relative_to(BOTCAD_ROOT))
        if rel not in self.EMITTER_PATHS:
            self.generic_visit(node)
            return

        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name == "rotation_between":
            self._warn(
                node,
                "Don't call rotation_between() directly in emitters. "
                "Use fastener_pose() or pose helpers from geometry.py instead.",
            )

        self.generic_visit(node)


@rule("no-hardcoded-identity-quat")
class NoHardcodedIdentityQuat(LintVisitor):
    """Ban hardcoded identity quaternions in emitter output.

    Lesson: viewer.py hardcoded quat=[1.0, 0.0, 0.0, 0.0] for mounted
    component bodies instead of computing the real orientation, making
    all mounted components appear unrotated regardless of face position.
    """

    EMITTER_PATHS = {"emit/viewer.py", "emit/mujoco.py"}

    def visit_Dict(self, node: ast.Dict) -> None:
        rel = str(self._path.relative_to(BOTCAD_ROOT))
        if rel not in self.EMITTER_PATHS:
            self.generic_visit(node)
            return

        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value != "quat":
                continue
            if self._is_identity_quat(value):
                self._warn(
                    value,
                    'Hardcoded identity quat for "quat" key. '
                    "Compute orientation from Placement.pose instead.",
                )

        self.generic_visit(node)

    @staticmethod
    def _is_identity_quat(node: ast.AST) -> bool:
        """Check if node is [1.0, 0.0, 0.0, 0.0] or (1, 0, 0, 0)."""
        if not isinstance(node, (ast.List, ast.Tuple)):
            return False
        elts = node.elts
        if len(elts) != 4:
            return False
        vals = []
        for e in elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, (int, float)):
                vals.append(float(e.value))
            else:
                return False
        return vals == [1.0, 0.0, 0.0, 0.0]


@rule("no-rotate-point-for-axes")
class NoRotatePointForAxes(LintVisitor):
    """Ban rotate_point() when applied to axis/direction vectors.

    Lesson: mount.rotate_point(mp.axis) was used to transform fastener
    insertion axes, but rotate_point() is a set of coordinate-swap lambdas
    designed for positions. Using it for direction vectors produced wrong
    orientations for camera and wheel fasteners.

    Flag any call to rotate_point() where the argument name contains
    'axis', 'dir', or 'normal'.
    """

    DIRECTION_HINTS = {"axis", "dir", "direction", "normal"}

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "rotate_point":
            for arg in node.args:
                if self._looks_like_direction(arg):
                    self._warn(
                        node,
                        "rotate_point() used on a direction/axis vector. "
                        "Use rotate_vec(quat, vec) or pose_transform_dir() instead.",
                    )
                    break

        self.generic_visit(node)

    def _looks_like_direction(self, node: ast.AST) -> bool:
        """Heuristic: does this argument look like a direction vector?"""
        # Check attribute access like mp.axis
        if isinstance(node, ast.Attribute):
            return any(h in node.attr.lower() for h in self.DIRECTION_HINTS)
        # Check variable name like fastener_axis
        if isinstance(node, ast.Name):
            return any(h in node.id.lower() for h in self.DIRECTION_HINTS)
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def lint_directory(root: Path) -> list[Violation]:
    """Lint all Python files under root."""
    violations: list[Violation] = []
    for py_file in sorted(root.rglob("*.py")):
        # Skip test files — they may legitimately use these patterns
        if "/tests/" in str(py_file) or py_file.name.startswith("test_"):
            continue
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        lines = source.splitlines()
        for v in _run_rules(py_file, tree):
            # Allow inline suppression: # plint: disable=rule-name
            # Check the violation line and surrounding lines (ruff may
            # reformat multi-line expressions, moving the comment)
            suppressed = False
            for offset in range(-1, 8):
                check_line = v.line + offset
                if 0 < check_line <= len(lines):
                    if f"plint: disable={v.rule}" in lines[check_line - 1]:
                        suppressed = True
                        break
            if not suppressed:
                violations.append(v)
    return violations


def main() -> int:
    violations = lint_directory(BOTCAD_ROOT)
    if not violations:
        return 0
    for v in violations:
        print(v, file=sys.stderr)
    print(f"\n{len(violations)} project lint violation(s) found.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
