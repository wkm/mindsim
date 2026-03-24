# Gremlin

Gremlin is a Claude Code skill that executes Linear issues as an AI agent. It picks up
well-scoped tickets, works in an isolated git worktree, and opens a draft PR when done.

## Label Workflow

Gremlin uses Linear labels to track each issue through its pipeline:

```
gremlin-ready          Human marks an issue as ready for Gremlin
       |
gremlin-working        Gremlin claims the issue and starts work
       |
       +---> gremlin-done     Success -- draft PR opened
       +---> gremlin-stuck    Bailed -- needs human input
```

The `gremlin` label is added permanently to every issue Gremlin touches, so you can
filter all Gremlin-related issues in Linear with `label:gremlin`.

## Usage

1. In Linear, add the `gremlin-ready` label to a well-scoped issue.
2. In Claude Code, invoke the skill:

```
/gremlin              # Show the queue of ready issues
/gremlin PER-42       # Execute a specific issue directly
```

Gremlin will:
- Claim the issue (set `gremlin-working`)
- Create a branch and worktree
- Implement the fix/feature
- Run `make validate`
- Commit, open a draft PR, and post the link to Linear
- Mark the issue `gremlin-done` (or `gremlin-stuck` if it bails)

## Writing Good Gremlin Issues

- Keep scope small: one file change or one well-defined feature
- Include clear acceptance criteria
- Link to relevant code paths or files when possible
- Ambiguous issues will be marked `gremlin-stuck` rather than guessed at

## More Information

Full design spec: `docs/superpowers/specs/2026-03-24-gremlin-linear-executor-design.md`
Skill definition: `.claude/skills/gremlin/SKILL.md`
