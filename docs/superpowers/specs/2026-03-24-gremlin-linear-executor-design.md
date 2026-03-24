# Gremlin: Linear Issue Executor for Claude Code

**Status:** Draft
**Date:** 2026-03-24
**Author:** wkm + Claude

## Overview

Gremlin is a Claude Code skill that picks up Linear issues and dispatches sub-agents to do the work. Each sub-agent runs in an isolated git worktree, posts progress to Linear, and opens a draft PR when done. A companion watcher script provides notifications via desktop alerts and Slack.

## Goals

- Turn well-scoped Linear tickets into draft PRs with minimal human intervention
- Provide heavy feedback into Linear so progress is visible without entering Claude Code
- Start small (interactive, single-issue) and build toward autonomous execution
- Keep infrastructure minimal — no webhook servers, no OAuth apps, no cloud hosting

## Non-Goals

- Fully autonomous operation (Phase 4 future work)
- Handling blocked/dependent ticket chains automatically
- Replacing human review — Gremlin opens draft PRs, never merges

## Architecture

Two components:

1. **`/gremlin` skill** — Claude Code skill that queries Linear, presents a queue, dispatches worktree sub-agents, and manages label/state transitions.
2. **`gremlin-watch` script** — Standalone Python script that polls Linear for Gremlin-labeled issues and sends notifications when state changes.

```
You (Claude Code session)
  │
  ├── /gremlin
  │     ├── Query: list_issues(label:"gremlin-ready", state:"Todo")
  │     ├── Present queue, you select issues
  │     └── For each selected issue:
  │           ├── Claim (labels + state)
  │           ├── Dispatch sub-agent (worktree, background)
  │           │     ├── Read issue context
  │           │     ├── Implement fix/feature
  │           │     ├── make validate
  │           │     ├── Commit + draft PR
  │           │     └── Post Linear comments (plan, PR link)
  │           └── On completion/failure: update labels + state
  │
  └── gremlin-watch.py (separate terminal tab)
        ├── Poll Linear every 60s for gremlin-* labels
        ├── Detect transitions (working→done, working→stuck)
        └── Notify via desktop + Slack
```

## Labels

Gremlin uses Linear labels to manage its pipeline. The `gremlin` label is permanent and enables filtering all Gremlin-touched issues.

| Label | Meaning | Set by | Removed when |
|---|---|---|---|
| `gremlin` | Gremlin touched this issue | Gremlin (on claim) | Never |
| `gremlin-ready` | Issue is ready for Gremlin | Human | Gremlin claims it |
| `gremlin-working` | Gremlin is actively working | Gremlin | Completion or bail |
| `gremlin-stuck` | Gremlin bailed, needs input | Gremlin | Human re-labels `gremlin-ready` |
| `gremlin-done` | Gremlin shipped a PR | Gremlin | Human merges/closes |

## Issue State Transitions

```
Todo + gremlin-ready
  │
  ├─ Gremlin claims ──→ In Progress + gremlin + gremlin-working
  │                            │
  │                  ┌─────────┴──────────┐
  │                  │                    │
  │            Succeeds              Bails
  │                  │                    │
  │                  ▼                    ▼
  │         In Review              Todo
  │         + gremlin              + gremlin
  │         + gremlin-done         + gremlin-stuck
  │         + PR link attached     + explanation comment
  │                  │                    │
  │                  ▼                    ▼
  │           Human reviews        Human provides input,
  │           and merges           re-labels gremlin-ready
  └────────────────────────────────────────┘
```

## `/gremlin` Skill

### Invocation

```
/gremlin              # Show queue, select issues to execute (Phase 1: pick one issue)
/gremlin MIN-42       # Execute a specific issue directly (Phase 1)
/gremlin status       # Show currently running Gremlin agents (Phase 2)
```

### Queue Discovery

```python
list_issues(label="gremlin-ready", state="Todo", orderBy="updatedAt")
```

Present a numbered list:

```
Gremlin Queue:
  1. [Urgent] MIN-42: Fix section colors in viewer
  2. [High]   MIN-51: Add chamfer preview to assembly tab
  3. [Normal] MIN-55: Bracket IR debug overlay

Execute which? (1,2,3 / all / none)
```

### Label Updates: Read-Modify-Write

The `save_issue` labels field is a **full replacement**, not additive. Every label update must:

1. `get_issue(id)` — read current labels
2. Compute the new label set (add/remove as needed)
3. `save_issue(id, labels=[...full list...])` — write complete set

This prevents accidentally stripping existing labels (priority, team, etc.). There is a theoretical race if a human edits labels concurrently, but this is acceptable for Phase 1 given manual invocation.

### Per-Issue Flow

1. **Fetch context:** `get_issue(id, includeRelations=true)` + `list_comments(issueId)`
2. **Claim:** Read current labels, then `save_issue(id, state="In Progress", labels=[...existing..., "gremlin", "gremlin-working"] minus "gremlin-ready")`
3. **Announce:** `save_comment(issueId, "Gremlin picking this up. Will post updates here.")`
4. **Dispatch:** Background sub-agent with `isolation: "worktree"` (see Sub-Agent Prompt below)
5. **Monitor:** Main session uses `TaskOutput` (blocking) to detect sub-agent completion
6. **On sub-agent completion:**
   - Success: Read labels, `save_issue(id, state="In Review", labels=[...+gremlin-done, -gremlin-working])` + comment with PR link
   - Failure: Read labels, `save_issue(id, state="Todo", labels=[...+gremlin-stuck, -gremlin-working])` + comment explaining what it needs

### Sub-Agent Timeout

Sub-agents have a **30-minute wall-clock timeout** for Phase 1. If a sub-agent has not completed within this window, the main session treats it as a bail: labels the issue `gremlin-stuck` and posts a timeout comment. This prevents runaway token consumption.

### Responsibility Split

| Concern | Owner |
|---|---|
| Reading issue context | Main session |
| Claiming/releasing issues (labels, state) | Main session |
| Creating branch, writing code, running tests | Sub-agent |
| Posting progress comments to Linear | Sub-agent (see note) |
| Committing, opening PR | Sub-agent |
| Final state transition (done/stuck) | Main session |

**Note on sub-agent Linear access:** MCP tool availability in worktree sub-agents needs to be verified in a Phase 1 spike. If Linear MCP tools are not accessible from worktree sub-agents, progress comments will be posted by the main session instead (sub-agent returns status via its output, main session relays to Linear). The main session remains the sole writer for label/state changes regardless.

This avoids race conditions — the main session is the single writer for label/state changes.

## Sub-Agent Prompt Template

```
You are Gremlin, an AI agent working on a Linear issue for the MindSim project.

## Issue: {issue_identifier} — {issue_title}

{issue_description}

## Discussion Context
{formatted_comments}

## Instructions

1. Create branch: `gremlin/{issue_identifier_lowercase}-{slugified_title}` (if branch exists from a prior attempt, reuse it)
2. Understand the issue — read relevant code, explore the codebase
3. Implement the fix/feature
4. Run `make validate` — lint, tests, and renders must pass
5. Commit with a descriptive message referencing the issue
6. Open a draft PR:
   - Title: "{issue_identifier}: {issue_title}"
   - Body: summary of changes, link to Linear issue
7. Post a comment on the Linear issue with the PR link

## Reporting

Post Linear comments at these milestones:
- When you have a plan (brief: what files, what approach)
- When the PR is ready (include link)
- If you cannot complete the work (unclear requirements, scope too large,
  blocked by something external), explain what you need and stop.
  Do NOT guess at ambiguous requirements.

## Technical Context

- This project uses `uv` (Python) and `pnpm` (TypeScript)
- Run `make validate` before committing
- Follow existing patterns in the codebase
- See CLAUDE.md for full conventions

## Issue ID: {issue_identifier}
Post comments via: save_comment(issueId: "{issue_identifier}", body: "...")
```

The sub-agent inherits the project's CLAUDE.md from the worktree, so conventions (ShapeScript, `.moved()` not `.locate()`, etc.) don't need repeating.

## `gremlin-watch` Script

### Purpose

Lightweight standalone Python script that polls Linear and sends notifications when Gremlin issues change state. Runs in a separate terminal tab, independent of Claude Code.

### Behavior

- Polls Linear every 60 seconds via the GraphQL API directly (not MCP — this runs outside Claude Code, so it needs its own API client)
- Maintains in-memory state: `{issue_id: last_seen_labels}`
- Detects label transitions and sends notifications:

| Transition | Notification |
|---|---|
| `gremlin-working` → `gremlin-done` | "Gremlin completed MIN-42: {title} -- PR: {url}" |
| `gremlin-working` → `gremlin-stuck` | "Gremlin stuck on MIN-42: {title} -- needs input" |
| New `gremlin-working` | Quiet log: "Gremlin started MIN-42" |

### Notification Channels

- **macOS desktop notification:** via `osascript` (always enabled)
- **Slack:** via incoming webhook URL (optional, from `SLACK_WEBHOOK_URL` env var)
- **Terminal:** log line to stdout (always)

### Configuration

```bash
# Required
export LINEAR_API_KEY=lin_...

# Optional
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
export GREMLIN_POLL_INTERVAL=60  # seconds, default 60
```

### Location

`scripts/gremlin-watch.py` — a utility, not part of the main application.

```bash
uv run python scripts/gremlin-watch.py
```

### Size

Target: ~80-120 lines. This is a notification pipe, not a decision-maker.

## Phased Rollout

### Phase 1: Manual Single-Issue (Prove the Loop)

**Goal:** Validate that Gremlin can take one well-scoped issue and produce a useful PR.

- Create all `gremlin-*` labels in Linear
- Build `/gremlin` skill with queue display and single-issue selection
- Sub-agent works in worktree, posts comments, opens draft PR
- Main session handles label transitions
- Test on one easy ticket (e.g., a viewer bug with clear repro)
- No watcher script — check Linear manually

**Success criteria:** Gremlin produces a PR that passes `make validate` and is mergeable with minor or no edits.

### Phase 2: Queue + Parallel Dispatch

**Goal:** Handle multiple issues in one session.

- Add queue discovery and selection UI
- Support dispatching multiple sub-agents in parallel (each in own worktree)
- Build `gremlin-watch.py` with desktop notifications

**Success criteria:** Dispatch 3 issues, all produce PRs, no worktree conflicts.

### Phase 3: Notifications + Polish

**Goal:** Smooth daily workflow integration.

- Add Slack integration to watcher
- Refine sub-agent prompt based on Phase 1-2 learnings
- Add `/gremlin status` to check running agents
- Tune error handling (better bail messages, retry guidance)

### Phase 4: Autonomous (Future)

**Goal:** Gremlin runs without manual invocation.

- Hook into Claude Code `/schedule` or external cron
- Potentially register as a Linear Agent via the Agent Interaction SDK
- Auto-pick up `gremlin-ready` issues
- Replace polling watcher with Linear webhook handler

## Prior Art

This design draws from:

- **[Cyrus](https://github.com/ceedaragents/cyrus):** Open-source Claude Code Linear agent. Webhook-driven, creates worktrees per issue, posts comments. Our Phase 4 target architecture.
- **[Jellypod claude-code-linear-agent](https://github.com/Jellypod-Inc/claude-code-linear-agent):** Extends Claude Code Action for Linear Agent SDK integration.
- **[Open SWE](https://github.com/langchain-ai/open-swe):** LangChain's multi-agent framework with Linear/Slack invocation and message queue middleware.
- **[Linear Agent Interaction SDK](https://linear.app/developers/agents):** First-class agent platform with delegation model and session lifecycle.
- **GitHub Copilot Coding Agent:** Assign issue → cloud agent → draft PR pattern.

## Phase 1 Spike: Verify Assumptions

Before building Phase 1, verify these with a quick test:

1. **MCP tools in worktree sub-agents:** Can a sub-agent dispatched with `isolation: "worktree"` call Linear MCP tools (`save_comment`, etc.)? If not, all Linear communication goes through the main session.
2. **`save_issue` label behavior:** Confirm that `labels` is a full replacement (not additive). Test the read-modify-write pattern.
3. **Sub-agent completion detection:** Confirm that `TaskOutput` with `block: true` works for detecting when a background worktree agent finishes.

## Open Questions

- Should Gremlin have a token/cost budget per issue? (Bail if exceeding threshold)
- Should the watcher script also monitor PR review status? (e.g., notify when CI passes on Gremlin PRs)
- How to handle issues that reference other issues as context? (Follow `relatedTo` links?)
- How should Gremlin handle issues with image attachments (screenshots, design mocks)? Can sub-agents meaningfully use image content from Linear?
