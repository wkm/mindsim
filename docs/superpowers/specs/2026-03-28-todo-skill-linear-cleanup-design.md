# `/todo` Skill + Linear Cleanup

## Summary

Create a `/todo` Claude Code skill for quick Linear issue creation, and clean up the Linear workspace to match current project reality.

## Linear Cleanup (one-time)

### Archive stale projects
- Paper Trading
- Signals
- Live Trading
- Experiment Queue

These are from a previous domain and no longer relevant to the mindsim/botcad work.
Set project state to `canceled`. Leave their issues as-is.

### Create area labels
Add workspace labels for cross-cutting filtering within BOB:
- `cad` — CAD pipeline, ShapeScript, build123d, geometry
- `viewer` — Three.js viewer, frontend, CSS
- `sim` — MuJoCo simulation, environments, training
- `training` — RL training, neural networks, policies

### Keep as-is
- **BOB** project remains the single active project
- Parent issues as epics (Botcad system, Wheeler bot, 2leg, Exploration environments)
- Existing labels (Bug, Feature, Improvement, gremlin-*)
- Status workflow (Backlog -> Todo -> In Progress -> In Review -> Done)

## `/todo` Skill

### Location
`.claude/skills/todo/SKILL.md`

### Two modes

**Quick mode** — when args are provided (`/todo fix coupler thickness`):
- Creates issue immediately in Perax team, BOB project, Backlog status
- Title = args verbatim
- Light label inference from title keywords:
  - "bug", "fix", "broken", "crash" -> Bug label
  - "feature", "add", "new" -> Feature label
  - "viewer", "css", "three", "frontend" -> `viewer` label
  - "cad", "mesh", "stl", "shapescript", "build123d" -> `cad` label
  - "sim", "mujoco", "environment" -> `sim` label
  - "train", "rl", "policy", "neural" -> `training` label
- Assigns to me
- Prints issue ID + URL

**Guided mode** — when no args (`/todo`):
- Ask for title
- Ask for priority (None / Urgent / High / Normal / Low — maps to API values 0-4)
- Show active parent issues and ask which one (if any) this belongs under
- Ask for labels (show available)
- Create and print result

### Implementation
Pure skill — no code. The skill prompt instructs Claude to use the Linear MCP tools (`save_issue`, `list_issues`) directly.
