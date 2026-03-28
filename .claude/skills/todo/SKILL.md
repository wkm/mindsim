---
name: todo
description: Create a Linear issue in the BOB project. With args, creates immediately. Without args, guides you through it.
argument-hint: [title]
disable-model-invocation: true
---

# /todo — Quick Linear Issue Creation

Create an issue in the **Perax** team, **BOB** project.

## Mode Selection

- If `$ARGUMENTS` is non-empty: use **Quick Mode**
- If `$ARGUMENTS` is empty: use **Guided Mode**

## Quick Mode

1. **Title** = `$ARGUMENTS` verbatim
2. **Infer labels** from the title (apply all that match):
   - Type labels (at most one):
     - "bug", "fix", "broken", "crash" → `Bug`
     - "feature", "add", "new" → `Feature`
     - Otherwise, no type label
   - Area labels (any number):
     - "viewer", "css", "three", "frontend" → `viewer`
     - "cad", "mesh", "stl", "shapescript", "build123d" → `cad`
     - "sim", "mujoco", "environment" → `sim`
     - "train", "rl", "policy", "neural" → `training`
3. **Create the issue** using the Linear MCP `save_issue` tool:
   - `team`: "Perax"
   - `project`: "BOB"
   - `title`: the title
   - `state`: "Backlog"
   - `assignee`: "me"
   - `labels`: inferred labels (omit if none matched)
4. **Print** the issue identifier and URL. Nothing else — keep it terse.

## Guided Mode

Ask these questions **one at a time**:

1. **Title** — "What's the issue?"
2. **Priority** — "Priority? (None / Urgent / High / Normal / Low)" — default None. Map to API values: None=0, Urgent=1, High=2, Normal=3, Low=4.
3. **Parent issue** — List active parent issues in BOB (use the Linear MCP `list_issues` tool filtered to BOB project with no parentId, state not Done/Canceled) and ask which one this belongs under, if any. Include "None" as an option.
4. **Labels** — Show available labels and ask which to apply. Include "None" as an option.

Then create the issue with all provided fields and print the identifier and URL.
