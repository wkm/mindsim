# Gremlin Phase 1: Manual Single-Issue Executor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that a Claude Code skill can take one Linear issue, dispatch a sub-agent in a worktree, and produce a draft PR — with status updates posted back to Linear.

**Architecture:** A Claude Code skill (`/gremlin`) reads a Linear issue, claims it via label updates, dispatches a background sub-agent in an isolated worktree, monitors for completion, and updates the Linear issue with results. No external infrastructure.

**Tech Stack:** Claude Code skills (markdown), Linear MCP tools, git worktrees, `gh` CLI for PRs

**Spec:** `docs/superpowers/specs/2026-03-24-gremlin-linear-executor-design.md`

---

## File Structure

```
.claude/skills/gremlin.md          — The /gremlin skill (main orchestrator)
scripts/gremlin-watch.py           — NOT in Phase 1 (Phase 2)
```

Phase 1 is a single file: the skill. Everything else is Claude Code infrastructure (MCP tools, Agent tool, worktrees) that already exists.

---

### Task 0: Spike — Verify Runtime Assumptions

**Files:**
- None (exploratory, no code committed)

Before building anything, verify the assumptions the design depends on.

- [ ] **Step 1: Test skill loading**

Create a trivial test skill to verify the directory convention and frontmatter format:

```bash
mkdir -p .claude/skills
```

Create `.claude/skills/hello.md`:
```markdown
---
name: hello
description: Test skill to verify skill loading works
user-invocable: true
---

Say "Hello from skill!" and stop.
```

Start a new Claude Code session and verify `/hello` appears in the skill list and loads correctly. Delete the test skill after.

- [ ] **Step 2: Test MCP tools in worktree sub-agent**

Dispatch a trivial sub-agent with `isolation: "worktree"` and have it call a Linear MCP tool (e.g., `list_teams`). Note the exact Agent tool parameters used and whether they work.

```
Agent(
  prompt: "Call the Linear MCP tool list_teams and return the result.",
  isolation: "worktree",
  subagent_type: "general-purpose"
)
```

Record: do MCP tools work in worktree sub-agents? If NO, the skill must relay all Linear comments through the main session.

- [ ] **Step 3: Test save_issue label behavior**

Pick any existing Linear issue. Call `get_issue` to read its current labels. Then call `save_issue` with `labels` set to a modified list. Verify:
- Does `labels` replace the full set, or append?
- After the test, restore the original labels.

Also test: does `save_issue` support a `links` parameter with format `[{url: "...", title: "..."}]`? This is needed for attaching PR URLs.

- [ ] **Step 4: Test sub-agent completion detection**

Dispatch a background sub-agent that does a trivial task (e.g., `echo done`). Then test which mechanism works for detecting completion:
- Try `TaskOutput(task_id, block: true, timeout: 30000)`
- If that doesn't work, try `TaskGet(task_id)` or other available tools

Record which tool and parameters work for blocking on background agent completion.

- [ ] **Step 5: Document spike results**

Record which assumptions held and which didn't:
- Skill frontmatter format: confirmed or adjusted
- MCP in worktree sub-agents: yes or no (determines comment posting strategy)
- `save_issue` labels: replace-all confirmed or not
- `save_issue` links: supported or not (determines PR link attachment method)
- Background agent completion: which tool/method works
- Agent tool exact parameter names: record what works

Update the spec and skill template if any assumptions were wrong.

---

### Task 1: Create Linear Labels

**Files:**
- None (Linear API calls only)

- [ ] **Step 1: Check existing labels**

```
list_issue_labels(team: "<your-team>")
```

Verify none of the gremlin labels already exist.

- [ ] **Step 2: Create labels**

Create each label via the Linear MCP tool:

```
create_issue_label(team: "<your-team>", name: "gremlin", color: "#7C3AED")
create_issue_label(team: "<your-team>", name: "gremlin-ready", color: "#10B981")
create_issue_label(team: "<your-team>", name: "gremlin-working", color: "#F59E0B")
create_issue_label(team: "<your-team>", name: "gremlin-stuck", color: "#EF4444")
create_issue_label(team: "<your-team>", name: "gremlin-done", color: "#3B82F6")
```

Colors: purple for parent, green=ready, amber=working, red=stuck, blue=done.

- [ ] **Step 3: Verify labels in Linear UI**

Open Linear and confirm all 5 labels appear under the team.

---

### Task 2: Write the `/gremlin` Skill

**Files:**
- Create: `.claude/skills/gremlin.md`

The skill is a markdown file with frontmatter that Claude Code loads when you type `/gremlin`. It contains instructions that the main session follows — it's not executable code, it's a prompt.

- [ ] **Step 1: Create the skills directory**

```bash
mkdir -p .claude/skills
```

- [ ] **Step 2: Write the skill file**

Create `.claude/skills/gremlin.md` with this content:

````markdown
---
name: gremlin
description: Pick up Linear issues labeled gremlin-ready and dispatch sub-agents to produce draft PRs
user-invocable: true
---

# Gremlin: Linear Issue Executor

You are the Gremlin orchestrator. Your job is to pick up Linear issues and dispatch sub-agents to do the work.

## Invocation

- `/gremlin` — show the queue of gremlin-ready issues, let user pick one
- `/gremlin ISSUE-ID` — execute a specific issue directly

## Step 1: Discover the Queue

Call the Linear MCP tool:

```
list_issues(label: "gremlin-ready", state: "Todo", orderBy: "updatedAt", limit: 20)
```

Present the results as a numbered list showing priority and title:

```
Gremlin Queue:
  1. [Urgent] MIN-42: Fix section colors in viewer
  2. [High]   MIN-51: Add chamfer preview to assembly tab
  3. [Normal] MIN-55: Bracket IR debug overlay

Pick an issue number to execute, or "none" to cancel.
```

If the user provided an ISSUE-ID argument, skip the queue and go directly to Step 2 with that issue.

If the queue is empty, say "No gremlin-ready issues found." and stop.

## Step 2: Fetch Full Context

For the selected issue:

```
get_issue(id: "<issue-identifier>", includeRelations: true)
list_comments(issueId: "<issue-identifier>")
```

Show the user a brief summary: title, description excerpt, number of comments, any blocking relations. Ask "Execute this issue?" for confirmation.

## Step 3: Claim the Issue

This is a read-modify-write operation because `save_issue` labels are a full replacement.

1. Read current labels from the `get_issue` result in Step 2
2. Compute new label set: add `gremlin` and `gremlin-working`, remove `gremlin-ready`
3. Update:

```
save_issue(id: "<issue-identifier>", state: "In Progress", labels: [<full computed list>])
save_comment(issueId: "<issue-identifier>", body: "Gremlin picking this up. Will post updates here.")
```

## Step 4: Dispatch Sub-Agent

Assemble the sub-agent prompt from the issue data, then dispatch:

```
Agent(
  description: "Gremlin: <issue-identifier> <short-title>",
  prompt: <assembled prompt — see template below>,
  isolation: "worktree",
  run_in_background: true,
  mode: "auto",
  name: "gremlin-<issue-identifier>"
)
```

Tell the user: "Dispatched gremlin agent for <issue-identifier>. Working in background."

## Step 5: Monitor and Finalize

Wait for the sub-agent to complete using TaskOutput with a 30-minute timeout:

```
TaskOutput(task_id: <agent-task-id>, block: true, timeout: 1800000)
```

When complete, read the sub-agent's output to determine success or failure.

**On success** (sub-agent opened a PR):
1. Read current labels from `get_issue`
2. Compute new labels: add `gremlin-done`, remove `gremlin-working`
3. Extract the PR URL from the sub-agent output
4. Update Linear:

```
save_issue(id: "<issue-identifier>", state: "In Review", labels: [<computed>], links: [{url: "<pr-url>", title: "Draft PR"}])
save_comment(issueId: "<issue-identifier>", body: "Gremlin completed this issue.\n\nPR: <pr-url>\n\n<brief summary from sub-agent>")
```

**On failure** (sub-agent bailed or timed out):
1. Read current labels from `get_issue`
2. Compute new labels: add `gremlin-stuck`, remove `gremlin-working`
3. Update Linear:

```
save_issue(id: "<issue-identifier>", state: "Todo", labels: [<computed>])
save_comment(issueId: "<issue-identifier>", body: "Gremlin could not complete this issue.\n\nReason: <failure reason from sub-agent output or 'Timed out after 30 minutes'>")
```

Report the outcome to the user.

**Worktree cleanup:** After finalization, check if the worktree sub-agent left behind a worktree directory. Claude Code auto-cleans worktrees with no changes, but successful runs will have committed changes on a branch. The branch is needed for the PR, so leave it. But if the sub-agent bailed with no changes, verify the worktree was cleaned up. If stale worktrees accumulate, clean them with `git worktree remove <path>`.

## Sub-Agent Prompt Template

Assemble this prompt, substituting values from the issue data:

```
You are Gremlin, an AI agent working on a Linear issue for the MindSim project.

## Issue: {issue_identifier} — {issue_title}

{issue_description}

## Discussion Context
{formatted_comments — or "No prior discussion." if empty}

## Instructions

1. Create a git branch: `gremlin/{issue_identifier_lowercase}-{slugified_title}`
   If the branch already exists from a prior attempt, check it out and continue from there.
2. Explore the codebase to understand the relevant code
3. Implement the fix or feature
4. Run `make validate` — lint, tests, and renders must pass
5. Commit with a descriptive message that references the issue identifier
6. Open a draft PR using `gh pr create`:
   - Title: "{issue_identifier}: {issue_title}"
   - Body: summary of what changed and why, plus "Linear: {issue_url}"
7. Post a comment on the Linear issue with the PR link:
   save_comment(issueId: "{issue_identifier}", body: "PR ready for review: <url>")

## Reporting via Linear Comments

Post comments on the Linear issue at these milestones:
- After exploring the code, post your plan (which files, what approach)
- When the PR is ready (include the link)
- If you cannot complete the work, explain what you need and STOP

Use: save_comment(issueId: "{issue_identifier}", body: "your message")

## Rules

- Do NOT guess at ambiguous requirements. If unclear, bail and explain.
- Do NOT make changes beyond the scope of this issue.
- Run `make validate` before committing. If it fails, attempt to fix the issues. If you cannot fix them after one retry, bail and explain the failures.
- Follow the conventions in CLAUDE.md.
```

If the spike (Task 0) showed that MCP tools do NOT work in worktree sub-agents, remove the "Post comments on the Linear issue" instructions from this template. Instead, add a note: "When you have a status update, include it clearly in your output text prefixed with LINEAR_COMMENT: so the orchestrator can relay it."
````

- [ ] **Step 3: Verify skill is discoverable**

Start a new Claude Code session in the mindsim project and check that `/gremlin` appears in the skill list. If using the Skill tool, verify `Skill(skill: "gremlin")` loads the skill content.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/gremlin.md
git commit -m "feat: add /gremlin skill — Linear issue executor (Phase 1)

Skill queries Linear for gremlin-ready issues, claims them via
label updates, dispatches worktree sub-agents to implement fixes,
and updates Linear with results (PR links or bail explanations)."
```

---

### Task 3: End-to-End Test on a Real Issue

**Files:**
- None (testing the skill against a real Linear issue)

- [ ] **Step 1: Create or pick a test issue**

Create a simple, well-scoped test issue in Linear. Something like a typo fix, a small refactor, or a minor viewer bug. Label it `gremlin-ready` and set state to `Todo`.

The issue should be completable in <5 minutes by the sub-agent so we can validate the full loop quickly.

- [ ] **Step 2: Run `/gremlin`**

Invoke `/gremlin` or `/gremlin <ISSUE-ID>`. Walk through the flow:
- Does the queue display correctly?
- Does claiming update the labels?
- Does the sub-agent dispatch successfully?

- [ ] **Step 3: Monitor the sub-agent**

Watch the sub-agent work. Check:
- Does it create a branch?
- Does it post progress comments to Linear?
- Does it run `make validate`?
- Does it open a draft PR?

- [ ] **Step 4: Verify Linear state after completion**

Check the Linear issue:
- Labels should be: `gremlin`, `gremlin-done`
- State should be: `In Review`
- There should be a PR link attached
- Comments should show the progression: claim, plan, PR ready

- [ ] **Step 5: Review the PR**

Check the draft PR:
- Does it reference the Linear issue?
- Does it pass CI / `make validate`?
- Is the code change reasonable?

- [ ] **Step 6: Document learnings**

Note what worked, what didn't, and what to adjust for Phase 2:
- Was the sub-agent prompt clear enough?
- Did the label transitions work correctly?
- How long did it take? Was the 30-min timeout reasonable?
- Any issues with worktree cleanup?

---

### Task 4: Test the Failure Path

**Files:**
- None (testing against a deliberately ambiguous issue)

- [ ] **Step 1: Create a vague test issue**

Create a Linear issue with an intentionally ambiguous description (e.g., "Make the viewer better"). Label it `gremlin-ready`.

- [ ] **Step 2: Run `/gremlin` on the vague issue**

The sub-agent should recognize it can't proceed and bail.

- [ ] **Step 3: Verify bail behavior**

Check:
- Labels should be: `gremlin`, `gremlin-stuck`
- State should be: `Todo`
- Comment should explain what Gremlin needs to proceed
- The worktree should be cleaned up (no leftover branch if no changes)

- [ ] **Step 4: Commit any skill adjustments**

If testing revealed issues with the skill, fix them and commit:

```bash
git add .claude/skills/gremlin.md
git commit -m "fix: refine gremlin skill based on Phase 1 testing"
```
