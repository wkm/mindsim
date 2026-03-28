# MindSim

Design robots from real components, train them in simulation, build the physical thing.

**Always use `uv` and `pnpm` and not `pip` and `npm`**

``sh
# 1. Make changes

# 2. Typecheck + lint (fast)
make lint

# 3. General test suite
make validate
```


## Principles

- Subagent use: Heavily rely on subagents to get information and keep the primary context clean. Subagents which do non-trivial code changes should use their own worktree.
- Make plans for any substantial changes. The plan should include the commit and testing strategy to validate changes. Validate changes yourself before saying something is done. As part of the final description of your work include a brief note on why you think the change is correct and how I can help validate it myself.
- Use the internet to research things and don't just rely on your training data. We're working in a new, rapidly evolving area.
- Simple is beautiful. Prefer composable modules. Prefer having one way to do things in the code. Complex operations should be easy to understand. Create intermediate representations that can be printed and debugged. Prefer declarative types that are operated on.
- The parametric skeleton is the **single source of truth**. One design produces everything — simulation, printable parts, assembly instructions, and training environments.
- **Derive from geometry, don't approximate.** When build123d/OCCT can compute a property (mass, COM, inertia, surface area) from actual CAD solid geometry, use it. The CAD solid is ground truth. Heuristic estimates are fallbacks, not the primary path.
- **Sim fidelity matters.** Geometry, mass, and actuation should match physical reality. CAD geometry = sim geometry — MuJoCo references the same STLs you'd send to a slicer.
- **One body, one mesh.** Each body solid is the union of its structural shell and all attachment geometry (brackets, cradles, couplers). That single mesh is both the collision and visual representation in simulation — no layered overlays. If geometry belongs to a body, union it into the body solid; don't bolt on a second visual-only mesh.
- **Design / Compose / Place / Cut.** The component geometry pipeline has four stages:
  1. **Design** — build each primitive (servo, bracket, coupler, clearance envelope) in its own local frame. The clearance/cut solid is a design-time artifact, visible in component renders.
  2. **Compose** — assemble related primitives (bracket wraps servo, coupler mates shaft) in local frame.
  3. **Place** — one rigid transform per component instance into the parent body frame.
  4. **Cut** — apply the pre-designed clearance solid to the parent body, using the **same transform** as Place.

  The critical invariant: Place and Cut use identical transforms. No axis-sign-dependent logic, no extra shifts at cut time. If a clearance shape needs asymmetry (e.g. outboard-only tolerance), that asymmetry is designed into the cut solid in step 1, in local frame, where it can be seen and validated.
- **Frozen dataclasses, explicit builders.** All `@dataclass` must use `frozen=True` (enforced by `tools/lint_project.py`). Things that accumulate state are plain classes named `FooBuilder`, not dataclasses. Builders produce frozen dataclasses: accumulate in local variables, construct the frozen result at the end. Suppress with `# plint: disable=frozen-dataclass` only for classes that genuinely need post-construction mutation (skeleton tree, lifecycle records).
- **Dimension types for physical quantities.** All physical quantities use `NewType` wrappers from `botcad/units.py` — never bare `float`. Use factory functions (`mm()`, `grams()`, `mpa()`) to convert from datasheet units. `Position` for spatial coordinates, `Size3D` for bounding box extents, `Vec3` for dimensionless directions. Dimensionless ratios and gains stay `float`. See `botcad/units.py` for the full list.
- **`.moved()`, never `.locate()`.** build123d's `.locate()` mutates in place and returns `self`. On `@lru_cache`d solids (bracket, envelope, coupler, servo), this silently corrupts the cached object for all future callers. Always use `.moved()` which creates an independent copy. The CAD steps debug viewer (`?cadsteps=bot:body`) makes this kind of bug visible.
- **Commit logs are a journal.** Explain _why_, not just _what_.
- Work with the user on improvement. Emit logs from interactions that will help you understand what a user did and what happened when they report a bug. Don't commit these logs.

## ShapeScript (Intermediate Representation)

The CAD pipeline uses ShapeScript, an intermediate representation between parametric design code and build123d/OCCT. It enables caching, swappable backends, and step-by-step visual debugging.

```bash
uv run mjpython main.py view --bot wheeler_arm              # Uses ShapeScript
uv run python main.py shapescript-debug --bot wheeler_arm --body base  # Step-by-step Rerun debugger
```

- ShapeScript is the **only** geometry path. There is no legacy fallback.
- **Debug:** `shapescript-debug` subcommand builds a body via ShapeScript and launches Rerun with per-op mesh snapshots.
- **Architecture:** `emit_body_ir()` emits typed ShapeScript ops (`botcad/shapescript/ops.py`). `OcctBackend` executes the ShapeScript against build123d.
- **Key files:** `botcad/shapescript/emit_body.py`, `botcad/shapescript/backend_occt.py`, `botcad/shapescript/program.py`, `botcad/shapescript/ops.py`

## TypeScript (Viewer)

The viewer (`viewer/`) is written in TypeScript. See **[TYPESCRIPT_STYLE.md](TYPESCRIPT_STYLE.md)** for the full style guide.

- **Biome** handles lint + format: `pnpm exec biome check --write viewer/`
- **tsc** handles type checking: `pnpm exec tsc --noEmit`
- Both run automatically via `make lint`
- Data model + sync pattern: `BotScene` (pure data) → `sync()` (maps to Three.js) → modes only mutate data

## Workflow Preferences

- **Subagent-centric**: Keep the main session clean and interactive. Delegate any non-trivial implementation, research, or multi-file changes to background subagents. The main session is for discussion, coordination, and quick answers. Subagents which change code get their own worktree.
- **Commit after every change**: Always commit with a descriptive message after completing a unit of work.
- **Parallel agents**: Launch multiple independent agents concurrently whenever possible. 
- **Build verification**: Agents should verify their changes before reporting completion.
- **Run `make validate` after every major step** — lint + tests + render regeneration. Review render diffs before committing.
- **Worktrees for experiments:** `make wt-new NAME=foo` → `exp/YYMMDD-foo` branch. Track in `EXPERIMENTS.md`.
- **Bot changes require `NEW_BOT_CHECKLIST.md`.**
- **TUI changes require snapshot tests:** `uv run pytest tests/test_tui_snapshots.py -v`
- **Merge** branches instead of rebasing. Be skeptical of cherry-picking from parallel work.
