"""Pure text formatter for Python repr strings.

Transforms single-line repr output into indented, multi-line format
for human readability. No formatting logic lives in data models —
this module is the single place where repr display formatting happens.
"""

from __future__ import annotations


def pretty_repr(s: str, indent: int = 2, line_width: int = 72) -> str:
    """Format a Python repr string with indentation.

    Short reprs (under *line_width* characters) are returned unchanged.
    Longer ones are broken across lines:
    - Break after opening ``(``
    - Break after each ``,`` at the current paren depth
    - Indent nested parens
    - Keep short inner expressions on one line

    Parameters
    ----------
    s:
        A single-line Python repr string.
    indent:
        Number of spaces per indentation level.
    line_width:
        Threshold below which a parenthesized group is kept on one line.
    """
    s = s.strip()
    if len(s) <= line_width:
        return s

    return _format_group(s, indent=indent, line_width=line_width, depth=0)


def _find_matching_paren(s: str, start: int) -> int:
    """Return the index of the closing paren matching the opener at *start*.

    Handles nested parens and skips characters inside string literals
    (single- and double-quoted).
    """
    open_ch = s[start]
    close_ch = {"(": ")", "[": "]", "{": "}"}[open_ch]
    depth = 1
    i = start + 1
    while i < len(s):
        ch = s[i]
        if ch in ("'", '"'):
            # Skip quoted string
            quote = ch
            i += 1
            while i < len(s) and s[i] != quote:
                if s[i] == "\\":
                    i += 1  # skip escaped char
                i += 1
            # i now on closing quote
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s) - 1  # fallback: end of string


def _split_top_level_args(s: str) -> list[str]:
    """Split a string on top-level commas, respecting parens and quotes."""
    args: list[str] = []
    current_start = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < len(s) and s[i] != quote:
                if s[i] == "\\":
                    i += 1
                i += 1
        elif ch in ("(", "[", "{"):
            i = _find_matching_paren(s, i)
        elif ch == ",":
            args.append(s[current_start : i + 1].strip())
            current_start = i + 1
        i += 1
    # Last arg (no trailing comma)
    tail = s[current_start:].strip()
    if tail:
        args.append(tail)
    return args


def _format_group(s: str, indent: int, line_width: int, depth: int) -> str:
    """Recursively format a repr string."""
    # Find the first opening paren
    paren_idx = -1
    for i, ch in enumerate(s):
        if ch in ("'", '"'):
            # Skip string literal — don't look for parens inside
            return s
        if ch in ("(", "[", "{"):
            paren_idx = i
            break

    if paren_idx == -1:
        # No parens — return as-is
        return s

    open_ch = s[paren_idx]
    close_ch = {"(": ")", "[": "]", "{": "}"}[open_ch]

    close_idx = _find_matching_paren(s, paren_idx)
    prefix = s[:paren_idx]
    inner = s[paren_idx + 1 : close_idx].strip()
    suffix = s[close_idx + 1 :].strip()

    # If the whole thing fits on one line, keep it compact
    full = f"{prefix}{open_ch}{inner}{close_ch}{suffix}"
    if len(full) <= line_width:
        return full

    # Split inner on top-level commas
    args = _split_top_level_args(inner)
    if not args:
        return full

    pad = " " * (indent * (depth + 1))
    close_pad = " " * (indent * depth)

    lines = [f"{prefix}{open_ch}"]
    for arg in args:
        # Strip trailing comma — we'll add our own
        a = arg.rstrip(",").strip()
        # Recursively format nested groups
        formatted = _format_group(
            a, indent=indent, line_width=line_width, depth=depth + 1
        )
        lines.append(f"{pad}{formatted},")
    lines.append(f"{close_pad}{close_ch}{suffix}")

    return "\n".join(lines)
