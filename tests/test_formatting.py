"""Tests for botcad.formatting.pretty_repr."""

from __future__ import annotations

from botcad.formatting import pretty_repr


def test_short_repr_unchanged():
    """Reprs under the line width threshold stay on one line."""
    s = "BodyId('base')"
    assert pretty_repr(s) == s


def test_nested_dataclass_repr():
    """AssemblyOp-style repr is broken across lines with nested short groups kept inline."""
    inp = (
        "AssemblyOp(step=0, action=AssemblyAction.INSERT, "
        "target=ComponentRef(body=BodyId('base'), mount_label='battery'), "
        "body=BodyId('base'), tool=ToolKind.FINGERS, "
        "approach_axis=(0, 0, -1), angle=None, "
        "prerequisites=(), description='Insert battery')"
    )
    result = pretty_repr(inp)

    # Should be multi-line
    lines = result.split("\n")
    assert len(lines) > 1, f"Expected multi-line output, got: {result!r}"

    # First line is the opener
    assert lines[0] == "AssemblyOp("

    # Last line is just the closing paren
    assert lines[-1] == ")"

    # Each field should appear on its own line
    assert any("step=0," in line for line in lines)
    assert any("description='Insert battery'," in line for line in lines)

    # Short nested groups stay on one line
    assert any(
        "target=ComponentRef(body=BodyId('base'), mount_label='battery')," in line
        for line in lines
    )


def test_no_parens():
    """Strings without parens are returned as-is."""
    s = "hello world this is a long string with no parentheses at all, really truly"
    assert pretty_repr(s) == s


def test_tuple_args():
    """Tuple arguments are kept on one line when short."""
    inp = "Foo(a=(1, 2, 3), b='x')"
    # Short enough to stay on one line
    assert pretty_repr(inp) == inp


def test_deeply_nested():
    """Multi-level nesting gets proper indentation."""
    inp = (
        "Outer(a=Middle(x=Inner(val=1), y=Inner(val=2)), "
        "b=Middle(x=Inner(val=3), y=Inner(val=4)))"
    )
    result = pretty_repr(inp)
    lines = result.split("\n")
    assert lines[0] == "Outer("
    # Inner groups should stay compact since they fit in 60 chars
    assert any("Middle(" in line for line in lines)


def test_trailing_comma_in_output():
    """Each argument line should have a trailing comma."""
    inp = (
        "AssemblyOp(step=0, action=AssemblyAction.INSERT, "
        "target=ComponentRef(body=BodyId('base'), mount_label='battery'), "
        "body=BodyId('base'), tool=None, approach_axis=None, "
        "angle=None, prerequisites=(), description='test')"
    )
    result = pretty_repr(inp)
    lines = result.split("\n")
    # All lines except the first (opener) and last (closer) should end with comma
    for line in lines[1:-1]:
        assert line.rstrip().endswith(","), f"Expected trailing comma: {line!r}"


def test_string_with_parens_not_split():
    """Parens inside string literals should not cause splitting."""
    s = "'hello (world)'"
    assert pretty_repr(s) == s


def test_empty_args():
    """Empty parens should be kept as-is."""
    s = "Foo()"
    assert pretty_repr(s) == s
