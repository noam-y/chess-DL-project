---
name: task_full_cycle
description: Execute full TaskVector implementation lifecycle for a specific task ID using developer, reviewer, and librarian flow, then commit and mark done. Use when user says //task_full_cycle SP-# or asks to run complete task cycle.
---

# Task Full Cycle

Run a complete task lifecycle for one TaskVector task:

1. Developer implementation
2. Reviewer validation
3. Librarian learnings processing
4. Commit (orchestrator)
5. Mark task done (orchestrator)

Use this skill when the user requests:
- `//task_full_cycle SP-#`
- "run the full cycle for SP-#"
- "developer reviewer librarian commit for SP-#"

## Input

- A single task id, usually `SP-#` (accept case-insensitive input like `sp-10`).

## Required Pre-step

From repo root, run:

```bash
./.taskvector/learnings/build_glossary.py
```

## Workflow

### 1) Validate task id and fetch task

```bash
tv task get SP-#
```

If missing/invalid, stop and ask user for correct ID.

### 2) Run developer agent

Spawn a developer subagent to implement the task.

Developer prompt requirements:
- Read `.cursor/skills/taskvector-load/SKILL.md`
- Run `tv task get SP-#`
- Run `./.taskvector/learnings/build_glossary.py`
- Set task `in_progress` with assignee
- Implement only that task
- Run relevant automated tests
- Return handoff with:
  - what changed
  - exact test commands and results
  - manual testing checklist (if needed)
  - risks/followups
  - learning file (if created)

Do **not** let developer mark task done or commit.

### 3) Run reviewer agent

Spawn reviewer subagent for the same task.

Reviewer must:
- run glossary + fetch task
- inspect diff against acceptance criteria
- verify test evidence
- return `APPROVED` or `NEEDS CHANGES`

If `NEEDS CHANGES`:
- resume the same developer agent with reviewer feedback
- rerun reviewer
- repeat until `APPROVED`

### 4) Run librarian agent

Spawn librarian to process `.taskvector/learnings/raw/`.

Expected outcome:
- case files updated if needed
- raw files cleaned up
- glossary rerun

### 5) Commit changes (orchestrator)

Before commit:
- exclude TaskVector runtime DB/WAL files from staging
- include actual code/test/docs/learnings changes

Commit message must be descriptive and include:
- task id (`SP-#`)
- intent ("why")

Use heredoc commit format.

### 6) Mark task done

After successful commit:

```bash
tv task done SP-#
```

## Safety Rules

- Run this flow for **one task at a time**.
- Do not parallelize mutating developer tasks.
- If developer handoff says manual user testing required, pause and ask user before commit.
- Never force-push or use destructive git operations.
- Do not mark done before commit.

## Output to User

After cycle completion, report:
- task id
- reviewer verdict
- tests run (summary)
- commit hash + message
- learnings updates (if any)
