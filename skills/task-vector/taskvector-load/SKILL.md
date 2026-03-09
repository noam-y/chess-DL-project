---
description: TaskVector integration for AI-assisted task management, MUST AUTO LOAD if user talks about task vector or asks about tasks.
---

# TaskVector Integration

TaskVector is a task management system designed for AI agents. Use the `tv` CLI to manage tasks.

## REQUIRED: Load Learnings First

**Before starting any task work, you MUST read the project learnings glossary.**

Run this exact command from the git repo root:
```bash
./.taskvector/learnings/build_glossary.py
```

This displays titles and one-line descriptions of each learning. If a specific learning seems relevant to your current task, read the full case file from `.taskvector/learnings/cases/`.

This is non-negotiable. Skipping learnings leads to repeated mistakes and wasted effort.

## Task Lifecycle

When working on a task, you MUST follow this workflow:

1. **Start the task (developer only)**: Developer claims task when starting work (status)
2. **Do the work**: Implement the required changes and run relevant tests/build
3. **Review**: Run reviewer after implementation, return to go back 2. if edits needed or 4.
4. **Complete the task (orchestrator only)**: Orchestrator marks `tv task done <ID>` after review is approved (and after commit, if applicable)

## Quick Reference

```bash
tv task list              # List all tasks in the current project
tv task list -s todo      # Filter by status
tv task get TV-1          # Get full task details
tv task start TV-1        # Mark task as in_progress
tv task done TV-1         # Mark task as done
```

## Task Identifiers

Tasks use the format `PROJECT-NUMBER`:
- `TV-1` = Task #1 in project "tv"
- `MYPROJ-42` = Task #42 in project "myproj"

Use `tv project list` to see available projects.

---

# Complete CLI Reference

---

## Commands Reference

### Projects

```bash
# List all projects
tv project list

# Create a project
tv project create "Project Name" --slug my-proj

# Get project details
tv project get my-proj

# Update project
tv project update my-proj --name "New Name"
```

### Tasks

```bash
# List tasks (uses default project if set)
tv task list
tv task list -p other-project
tv task list -s todo                    # Filter by status
tv task list -s in_progress --assignee claude

# Create task
tv task create "Task title"
tv task create "Task title" -p my-proj --priority high
tv task create "Task title" -b "Description body"

# Get task details
tv task get TV-1
tv task get my-proj-1

# Update task
tv task update TV-1 --title "New title"
tv task update TV-1 -s in_progress
tv task update TV-1 --assignee alice --priority low

# Status updates
tv task start TV-1                      # Set status to in_progress
tv task done TV-1                       # Set status to done

# Subtasks
tv task create "Subtask title" --parent TV-1

# Dependencies
tv task block TV-2 --by TV-1            # TV-2 is blocked by TV-1
tv task unblock TV-2 --by TV-1          # Remove dependency

# Delete task
tv task delete TV-1
```

### Events

```bash
# View recent events
tv events
tv events -p my-proj                    # Filter by project
tv events --limit 50                    # More events
```

### Servers

```bash
# Start API server (default port 3000)
tv serve
tv serve -p 8080

# Open WebUI in browser
tv webui
```

### Configuration

```bash
# Show current config
tv config
```

---

## Status Values

| Status | Description |
|--------|-------------|
| `backlog` | Not yet prioritized |
| `todo` | Ready to work on |
| `in_progress` | Currently being worked |
| `done` | Completed |

---

## Priority Values

| Priority | Description |
|----------|-------------|
| `high` | Urgent |
| `medium` | Normal (default) |
| `low` | Can wait |

---

## Multiline Input

### Heredoc (recommended for scripts)

```bash
tv task create "Add feature" -b "$(cat <<'EOF'
## Context
Background information here.

## Requirements
- First requirement
- Second requirement

## Acceptance criteria
- [ ] Tests pass
- [ ] Feature works
EOF
)"
```

### From file

```bash
tv task create "Add feature" -b @requirements.md
```

### Pipe from stdin

```bash
echo "Task description" | tv task create "Title" -b -
```

---

## Output Formats

### Markdown (default)

```bash
tv task get TV-1
```

```markdown
## Task: TV-1

**Title:** Implement feature X
**Status:** in_progress | **Priority:** high
**Project:** tv | **Assignee:** claude

### Description
Full task description here...
```

### JSON

```bash
tv task get TV-1 --json
```

```json
{
  "success": true,
  "data": {
    "identifier": "TV-1",
    "title": "Implement feature X",
    "status": "in_progress"
  }
}
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Authentication error |
| 4 | Resource not found |
| 5 | Validation error |

---

## Best Practices (Direct CLI Usage)

**Note:** These practices apply when YOU (the orchestrator) work directly in the CLI. For spawned subagents (developer/reviewer/librarian), they follow their own agent-specific protocols defined in their agent files.

1. Always run `tv task start <ID>` before beginning work on a task
2. Read the full task body with `tv task get <ID>` to understand requirements
3. Mark tasks complete with `tv task done <ID>` when finished
4. Create subtasks with `tv task create "Title" --parent <ID>` for complex work

---

# Agent Orchestration

You (the main Claude instance) are the orchestrator. You interact with the user and either follow skills directly or spawn agents for specific phases. **Never start implementation without user approval.**

> **IMPORTANT: Summarize Your Actions**
>
> After completing a workflow (creating tasks, implementing, reviewing), briefly summarize what you did:
> - "Followed plan skill → created TV-28, TV-29, TV-30"
> - "Used developer for TV-28 → completed, then reviewer → approved"
> - "Developer blocked → explored more → enriched task → resumed developer → done"

---

## Planning New Features (Plan Skill)

> **CRITICAL: New Features Use the Plan Skill**
>
> When a user asks to **build something new**, **add a feature**, or **create tasks**:
>
> 1. **Read the plan skill**: `.claude/skills/taskvector-plan/SKILL.md` if not yet
> 2. **Follow its phases** directly (you drive the conversation, not a subagent)
>
> **Only skip the plan skill** if the user explicitly says "add a quick task for X" or provides a fully detailed task spec.

The plan skill guides you through:

1. **UNDERSTAND** - Ask 2-3+ clarifying questions
2. **EXPLORE** - Spawn explore agents to learn patterns
3. **OPTIONS** - Present 2-3 implementation approaches with pros/cons
4. **DEEP DIVE** - Explore the chosen approach in detail
5. **CREATE TASKS** - Build junior-dev-ready tasks in TaskVector

This interactive approach enables richer conversation than spawning an agent.

---

## Implementing Tasks (Developer Agent)

> **IMPORTANT: Delegate Implementation to Developer Agent**
>
> When a user asks you to implement a task (TV-X), **always spawn `taskvector-developer`** to do the work. Do NOT write the code yourself unless the user explicitly asks you to implement it directly.

```
Task(
  description="Implement TV-X",
  subagent_type="taskvector-developer",
  prompt="Implement task TV-X. Run `tv task get TV-X` to get full requirements, then `tv task start TV-X` before implementing. Hand off to orchestrator when complete - do NOT run `tv task done TV-X`."
)
```

**Developer agent workflow:**
1. Runs `tv task get TV-X` to fetch requirements
2. Runs `tv task start TV-X` to claim the task
3. Implements changes following the task spec
4. Returns: "Completed" + summary, "Blocked" + reason, or "Failed" + error

---

## Reviewing Changes (Reviewer Agent)

After implementation, spawn the reviewer:

```
Task(
  description="Review TV-X",
  subagent_type="taskvector-reviewer",
  prompt="Review implementation for TV-X. Run `tv task get TV-X` to get requirements, then verify the changes meet acceptance criteria."
)
```

**Reviewer returns:**
- "Approved" - ready to commit
- "Needs changes: [issues]" - resume developer to fix

**After reviewer approval:**
Check the developer's handoff for "Manual testing required for user" section. If present, ASK the user to run those tests before committing. Do NOT skip this step - automated tests don't catch all integration issues.

---

## Available Agents

| Agent | Use When |
|-------|----------|
| `explore` | Need to investigate codebase patterns (used during architect skill) |
| `taskvector-developer` | Need to implement a specific task (TV-X) |
| `taskvector-reviewer` | Need to review implementation for bugs, quality, conventions |
| `taskvector-librarian` | Need to process raw learnings in `.taskvector/learnings/raw/` |

---

## Feature Development Flow

```
User Request
    │
    ├─► Read plan skill (.claude/skills/taskvector-plan/SKILL.md)
    │
    ├─► PHASE 1: Ask 2-3+ clarifying questions
    │
    ├─► PHASE 2: Spawn explore agents to understand patterns
    │       └─► (CAN run multiple explore agents in parallel - read-only)
    │
    ├─► PHASE 3: Present 2-3 implementation options
    │       └─► Compare pros/cons, effort, scope
    │       └─► User chooses approach
    │
    ├─► PHASE 4: Deep dive on chosen approach
    │       └─► Spawn explore agent for specific details
    │
    ├─► PHASE 5: Create tasks via `tv task create`
    │
    ├─► Present tasks to user: `tv task list`
    │
    ├─► **WAIT for user to choose which task(s) to implement**
    │
    ├─► For each task (TV-X) in order (**SEQUENTIAL, one at a time**):
    │       │
    │       ├─► Spawn taskvector-developer with TV-X
    │       │       └─► **WAIT** for completion before next task
    │       │       └─► Returns: "Completed" + agent_id / "Blocked" / "Failed"
    │       │
    │       ├─► If blocked: explore more, enrich task, resume developer
    │       │
    │       ├─► Spawn taskvector-reviewer with TV-X
    │       │       └─► Returns: "Approved" / "Needs changes: [issues]"
    │       │
    │       ├─► If needs changes:
    │       │       └─► Resume taskvector-developer with feedback
    │       │
    │       ├─► If approved:
    │       │       ├─► Check developer handoff for "Manual testing required"
    │       │       ├─► If manual tests listed: ASK USER to run them, wait for confirmation
    │       │       ├─► After user confirms (or no manual tests): commit changes
    │       │       └─► tv task done TV-X (after commit!)
    │       │
    │       └─► **THEN proceed to next task**
    │
    └─► Present results to user
```

---

## Resuming Agents

Every agent returns an `agentId` when it completes. Use the `resume` parameter to continue with full context preserved.

```
# Initial spawn
Task(prompt="Implement TV-42", subagent_type="taskvector-developer")
→ Returns: "Completed: added auth middleware" + agentId: "abc123"

# Code review finds issues
Task(prompt="Review TV-42", subagent_type="taskvector-reviewer")
→ Returns: "Needs changes: missing error handling"

# Resume original developer (keeps context)
Task(prompt="Fix: missing error handling",
     subagent_type="taskvector-developer",
     resume="abc123")
```

| Situation | Action |
|-----------|--------|
| Fixing review feedback | Resume the developer agent |
| Developer blocked | Explore more, update task, resume developer |
| Unrelated new task | Spawn fresh agent |

---

## Critical Rules

1. **New features use plan skill** - read and follow `.claude/skills/taskvector-plan/SKILL.md`
2. **Delegate implementation to developer agent** - never write task code yourself unless explicitly asked
3. **Never auto-implement** - always wait for user to say which tasks to work on
4. **Review after implementation** - spawn reviewer before marking done
5. **Resume for fixes** - resume the developer (don't spawn new) when fixing issues
6. **Orchestrator commits** - commit after approval, not during implementation
7. **Sequential implementation** - run developer agents one at a time (see Parallelization Rules below)
8. **Manual testing before commit** - if developer handoff includes "Manual testing required for user", ASK the user to run those tests and wait for confirmation before committing

---

## Parallelization Rules

> **CRITICAL: Only parallelize read-only operations. Mutating agents run sequentially.**

### CAN run in parallel (read-only):
- Multiple `explore` agents investigating different parts of codebase
- `explore` + `taskvector-reviewer` (reviewer only reads, doesn't modify)

### MUST run sequentially (mutating):
- `taskvector-developer` agents - **always one at a time**
- Tasks that modify the same files (even if different tasks)
- Tasks with dependencies (e.g., TV-88 before TV-89 if both touch Header.tsx)

### Why sequential for developers?
1. **File conflicts** - parallel developers may overwrite each other's changes
2. **Missing context** - second developer doesn't see first developer's work
3. **Review confusion** - harder to attribute changes to specific tasks
4. **Learning quality** - developers create better learnings with full context

### Correct workflow:
```
# Good: Sequential developers
Developer TV-87 → wait → Reviewer TV-87 → Developer TV-88 → wait → Reviewer TV-88

# Good: Parallel exploration
Explore "auth patterns" + Explore "db patterns" (both read-only)

# Bad: Parallel developers (DO NOT DO THIS)
Developer TV-87 + Developer TV-88 (both modify files)
```

### When user provides recommended order:
Follow it exactly. The user knows which tasks share files or have dependencies. But escalate if you think the user is missing something.

---

## Subagent Response Convention

Subagents should return one of:
- **Success**: `"Completed: [summary of changes]"`
- **Blocked**: `"Blocked: [what's unclear or needed]"`
- **Failed**: `"Failed: [error details]"`

---

## Learnings System

Developer agents write learnings directly to `.taskvector/learnings/raw/<topic>_<slug>.md` and mention the filename in their handoff.

**File format:**
```markdown
---
description: "Brief one-line summary with #hashtags for topics"
learned: "YYYY-MM-DD_HH:MM:SS"
---

# Title

**Context:** When/where this applies

[Explanation of the learning]

## Example (if applicable)

[Code snippet or demonstration]
```

**Topics:** `pattern`, `preference`, `codebase`, `gotcha`, `build-system`

**To organize learnings** (user-initiated), spawn the librarian:
```
Task(prompt="Process new learnings in .taskvector/learnings/raw/",
     subagent_type="taskvector-librarian")
```

---

## Handling Blocked Developers

When a developer agent returns "Blocked":

1. **Explore more** - spawn explore agent to research the blocker
2. **Update the task** - add context via `tv task update TV-X -b "[enriched body]"`
4. comment on the task with 
 `tv task comment add -b "updated task with..." <taskid>`
3. **Resume the developer** with the enriched task

```
Task(prompt="Re-read TV-X, task has been updated with more context",
     subagent_type="taskvector-developer",
     resume="[developer-agent-id]")
```
