# Ruff Lint Uplift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade ruff config from 4 rule sets to 12+, add build123d import guardrails, and fix all resulting violations.

**Architecture:** Update `pyproject.toml` ruff config to match quality level of Ruff's own codebase. Add `TID` banned-api to enforce that only authorized modules import `build123d`/`OCP` directly. Fix all violations in-place — auto-fix where safe, manual fix for the rest.

**Tech Stack:** ruff (linter/formatter), pyproject.toml config

---

## Context for implementers

### What we're adding and why

| Rule set | What it catches | Why |
|---|---|---|
| `B` (bugbear) | Mutable defaults, bare except, raise-without-from | Enabled by 5/5 surveyed projects |
| `C4` (comprehensions) | Unnecessary list/dict constructors | 4/5 projects |
| `PIE` (pie) | Unnecessary pass, dict unpacking | 4/5 projects |
| `SIM` (simplify) | Yoda conditions, collapsible ifs, reimplemented builtins | Polars + Ruff |
| `RUF` (ruff-specific) | Unused noqa, mutable class defaults, unused unpacking | Polars + Ruff |
| `PERF` (perflint) | Manual list comprehension, dict iterator misuse | Pydantic + Polars |
| `T10` (debugger) | Leftover `breakpoint()` / `import pdb` | Basic hygiene |
| `TID` (tidy-imports) | Banned API enforcement (build123d containment) | Project-specific |

### What we're ignoring and why

| Rule | Why ignore |
|---|---|
| `E501` | Formatter handles line length |
| `RUF001/002/003` | Ambiguous unicode chars — we use degree symbols (°) legitimately |
| `SIM108` | if/else vs ternary — readability preference (Polars ignores too) |
| `PLC0415` | Import-outside-top-level — codebase uses lazy imports deliberately (640 violations) |

### build123d containment strategy

Ban `build123d` and `OCP` globally via `TID251` banned-api, then whitelist authorized files via `per-file-ignores`. Authorized consumers:

| File | Why allowed |
|---|---|
| `botcad/shapescript/backend_occt.py` | **Primary authorized consumer** — executes ShapeScript IR |
| `botcad/emit/cad.py` | Legacy geometry construction — Part B will migrate to ShapeScript |
| `botcad/fasteners.py` | Legacy geometry — Part B will migrate |
| `botcad/connectors.py` | Legacy geometry — Part B will migrate |
| `botcad/components/*.py` | Legacy geometry — Part B will migrate |
| `botcad/cad_utils.py` | Type coercion utilities (reads shapes, doesn't construct) |
| `botcad/clearance.py` | Distance queries (reads shapes, doesn't construct) |
| `botcad/render_svg.py` | SVG rendering (reads shapes, doesn't construct) |
| `botcad/debug_drawing.py` | Debug visualization |
| `botcad/validation.py` | Export/validation |
| `botcad/emit/component_renders.py` | STL export for renders |
| `botcad/emit/drawings.py` | Drawing generation |
| `botcad/shapescript/debug_rerun.py` | Debug visualization |
| `tests/**` | Tests can import directly |
| `scripts/**` | Scripts can import directly |

### Violation summary (164 total)

**Auto-fixable (50):** `RUF100` (32), `SIM300` (9), `C420` (3), `RUF022` (3), `SIM114` (3)

**Manual fix (114):**
- `RUF059` unused-unpacked-variable (45) — replace with `_`
- `PERF401` manual-list-comprehension (19) — rewrite as comprehension
- `B904` raise-without-from (16) — add `from err` / `from None`
- `B905` zip-without-strict (12) — add `strict=True/False`
- `B007` unused-loop-variable (6) — rename to `_`
- Remaining (16) — small one-off fixes

---

### Task 1: Update ruff config in pyproject.toml

**Files:**
- Modify: `pyproject.toml:46-58`

- [ ] **Step 1: Update the ruff config**

Replace the `[tool.ruff.lint]` section with the expanded config including new rule sets, ignores, TID banned-api, and per-file-ignores.

```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "PIE",    # flake8-pie
    "SIM",    # flake8-simplify
    "RUF",    # ruff-specific rules
    "PERF",   # perflint
    "T10",    # flake8-debugger
    "TID",    # flake8-tidy-imports
]
ignore = [
    "E501",    # line too long — formatter handles it
    "RUF001",  # ambiguous unicode in strings (degree symbols etc)
    "RUF002",  # ambiguous unicode in docstrings
    "RUF003",  # ambiguous unicode in comments
    "SIM108",  # if-else vs ternary — readability preference
]

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"build123d".msg = "Do not import build123d directly. Use ShapeScript IR (botcad/shapescript/). Only backend_occt.py should call build123d."
"OCP".msg = "Do not import OCP directly. Use ShapeScript IR (botcad/shapescript/). Only backend_occt.py should call OCP/OCCT."

[tool.ruff.lint.per-file-ignores]
# build123d/OCP authorized consumers (TID251)
"botcad/shapescript/backend_occt.py" = ["TID251"]
"botcad/shapescript/debug_rerun.py" = ["TID251"]
"botcad/emit/cad.py" = ["TID251"]
"botcad/emit/component_renders.py" = ["TID251"]
"botcad/emit/drawings.py" = ["TID251"]
"botcad/fasteners.py" = ["TID251"]
"botcad/connectors.py" = ["TID251"]
"botcad/components/*.py" = ["TID251"]
"botcad/cad_utils.py" = ["TID251"]
"botcad/clearance.py" = ["TID251"]
"botcad/render_svg.py" = ["TID251"]
"botcad/debug_drawing.py" = ["TID251"]
"botcad/validation.py" = ["TID251"]
"tests/**" = ["TID251"]
"scripts/**" = ["TID251"]
```

- [ ] **Step 2: Verify ruff parses the config**

Run: `uv run ruff check --statistics botcad/ mindsim/ tests/ scripts/ 2>&1 | tail -30`

Expected: ~164 violations from the new rules (not config parse errors). The TID251 violations should be zero (all legitimate consumers whitelisted). If any TID251 violations appear, add the file to per-file-ignores.

- [ ] **Step 3: Commit config change**

```bash
git add pyproject.toml
git commit -m "chore: expand ruff lint rules — add B, C4, PIE, SIM, RUF, PERF, T10, TID

Add build123d/OCP import guardrails via TID251 banned-api.
Whitelisted files that legitimately use build123d will be migrated
to ShapeScript IR in a follow-up branch.

Rule selection modeled after Ruff's own codebase + Polars."
```

---

### Task 2: Auto-fix safe violations

**Files:** Multiple (ruff will modify in-place)

- [ ] **Step 1: Run ruff auto-fix**

```bash
uv run ruff check --fix botcad/ mindsim/ tests/ scripts/
```

This fixes ~50 violations:
- `RUF100` — removes unused `# noqa` comments (32)
- `SIM300` — fixes yoda conditions like `0 == x` → `x == 0` (9)
- `C420` — simplifies dict comprehensions (3)
- `RUF022` — sorts `__all__` (3)
- `SIM114` — merges if-branches with same body (3)

- [ ] **Step 2: Review the diff**

```bash
git diff --stat
git diff
```

Scan the diff to make sure auto-fixes are correct. Pay special attention to `SIM114` (merged if-branches) — verify the conditions are truly equivalent.

- [ ] **Step 3: Run lint to confirm reduction**

```bash
uv run ruff check --statistics botcad/ mindsim/ tests/ scripts/
```

Expected: ~114 remaining violations (all manual-fix categories).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: apply ruff auto-fixes for RUF100, SIM300, C420, RUF022, SIM114"
```

---

### Task 3: Fix RUF059 — unused unpacked variables (45 violations)

**Files:** Multiple across botcad/

- [ ] **Step 1: List all RUF059 violations**

```bash
uv run ruff check --select RUF059 botcad/ mindsim/ tests/ scripts/
```

- [ ] **Step 2: Fix each violation**

For each violation, replace the unused unpacked variable with `_`:

```python
# Before:
x, y, z = vec3
# (y is unused)

# After:
x, _, z = vec3
```

If multiple variables are unused, use `_` for each. If only the last N are unused, consider slicing instead.

**Important:** Some unpacking may be intentional for readability (e.g., `x, y, z = position` where only `x` and `z` are used but `y` documents the structure). Use `_y` instead of `_` in those cases to suppress the lint while preserving documentation.

- [ ] **Step 3: Run lint to confirm fixes**

```bash
uv run ruff check --select RUF059 botcad/ mindsim/ tests/ scripts/
```

Expected: 0 violations.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: replace unused unpacked variables with _ (RUF059)"
```

---

### Task 4: Fix PERF401 — manual list comprehension (19 violations)

**Files:** Multiple across botcad/

- [ ] **Step 1: List all PERF401 violations**

```bash
uv run ruff check --select PERF401 botcad/ mindsim/ tests/ scripts/
```

- [ ] **Step 2: Fix each violation**

Convert manual list-building loops to comprehensions:

```python
# Before:
result = []
for item in items:
    result.append(transform(item))

# After:
result = [transform(item) for item in items]
```

**Caution:** Some loops may have side effects or complex conditionals that don't cleanly convert to comprehensions. If a loop body does more than a single append, leave it as-is and add `# noqa: PERF401` with a comment explaining why.

- [ ] **Step 3: Run lint to confirm fixes**

```bash
uv run ruff check --select PERF401 botcad/ mindsim/ tests/ scripts/
```

Expected: 0 violations.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: convert manual list-building loops to comprehensions (PERF401)"
```

---

### Task 5: Fix B904 — raise without from (16 violations)

**Files:** Multiple across botcad/

- [ ] **Step 1: List all B904 violations**

```bash
uv run ruff check --select B904 botcad/ mindsim/ tests/ scripts/
```

- [ ] **Step 2: Fix each violation**

Add exception chaining to `raise` statements inside `except` blocks:

```python
# Before:
try:
    something()
except SomeError:
    raise DifferentError("msg")

# After (preserve original traceback):
try:
    something()
except SomeError as err:
    raise DifferentError("msg") from err

# Or (intentionally suppress original — use sparingly):
try:
    something()
except SomeError:
    raise DifferentError("msg") from None
```

Use `from err` by default. Only use `from None` when the original exception is genuinely irrelevant (e.g., converting a KeyError to a domain-specific "not found" error where the dict internals don't help the caller).

- [ ] **Step 3: Run lint to confirm fixes**

```bash
uv run ruff check --select B904 botcad/ mindsim/ tests/ scripts/
```

Expected: 0 violations.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: add exception chaining to raise-in-except (B904)"
```

---

### Task 6: Fix B905 — zip without strict (12 violations)

**Files:** Multiple across botcad/

- [ ] **Step 1: List all B905 violations**

```bash
uv run ruff check --select B905 botcad/ mindsim/ tests/ scripts/
```

- [ ] **Step 2: Fix each violation**

Add `strict=` parameter to `zip()` calls:

```python
# Before:
for a, b in zip(list_a, list_b):

# After (if lengths MUST match — use this by default):
for a, b in zip(list_a, list_b, strict=True):

# After (if lengths may differ intentionally):
for a, b in zip(list_a, list_b, strict=False):
```

**Decision rule:** Use `strict=True` unless the code clearly handles or expects different-length iterables (e.g., zip-truncation is the intended behavior). Read the surrounding code to decide.

- [ ] **Step 3: Run lint to confirm fixes**

```bash
uv run ruff check --select B905 botcad/ mindsim/ tests/ scripts/
```

Expected: 0 violations.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: add strict= to zip() calls (B905)"
```

---

### Task 7: Fix remaining violations (~16 misc)

**Files:** Multiple across botcad/

- [ ] **Step 1: List all remaining violations**

```bash
uv run ruff check botcad/ mindsim/ tests/ scripts/
```

- [ ] **Step 2: Fix each violation by rule**

**B007** (unused-loop-variable, 6): Rename to `_`:
```python
for _ in range(n):  # was: for i in range(n)
```

**SIM110** (reimplemented-builtin, 3): Replace with `any()`/`all()`:
```python
# Before:
for item in items:
    if predicate(item):
        return True
return False
# After:
return any(predicate(item) for item in items)
```

**PIE810** (multiple-starts-ends-with, 2): Combine into tuple arg:
```python
# Before:
s.startswith("a") or s.startswith("b")
# After:
s.startswith(("a", "b"))
```

**SIM102** (collapsible-if, 2): Merge nested ifs:
```python
# Before:
if a:
    if b:
        do_thing()
# After:
if a and b:
    do_thing()
```

**Remaining** (B011, B023, C408, PERF102, RUF005, RUF012, SIM103, SIM105, SIM118 — 1 each): Fix individually based on the specific violation. Read the code around each violation before fixing.

- [ ] **Step 3: Run full lint to confirm zero violations**

```bash
uv run ruff check botcad/ mindsim/ tests/ scripts/
```

Expected: 0 violations (or only pre-existing E/F/I/UP violations that were already present).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "fix: resolve remaining ruff violations (B007, SIM110, PIE810, SIM102, misc)"
```

---

### Task 8: Full validation

- [ ] **Step 1: Run make lint**

```bash
make lint
```

Expected: Clean pass (ruff + biome + tsc).

- [ ] **Step 2: Run make validate**

```bash
make validate
```

Expected: All tests pass. No regressions from the lint fixes.

- [ ] **Step 3: Final commit if any formatting changes**

If `make lint` auto-formatted anything:

```bash
git add -A
git commit -m "chore: apply formatter after lint rule fixes"
```
