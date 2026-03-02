---
description: TaskVector CLI only (without agentic enhancement) description for task management
---

# TaskVector CLI

TaskVector is a task management system. Use the `tv` CLI to manage tasks.

## Task Lifecycle

When working on a task, follow this workflow:

1. **Start the task**: Run `tv task start <ID>` to mark it as in_progress
2. **Do the work**: Implement the required changes
3. **Complete the task**: Run `tv task done <ID>` to mark it as done

## Quick Reference

```bash
tv task list              # List all tasks
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

## Learnings Glossary

Learnings glossary is generated on demand from frontmatter descriptions.

Run this from the git repo root:
```bash
./.taskvector/learnings/build_glossary.py
```

Frontmatter requirement (per learning file):
```
---
description: "One-line summary with #tags"
learned: "YYYY-MM-DD_HH:MM:SS"
---
```

## Commands

### Tasks

```bash
tv task list                           # List tasks
tv task list -p project -s todo        # Filter by project and status
tv task create "Title"                 # Create task
tv task create "Title" -b "Body"       # With description
tv task get TV-1                       # Get details
tv task update TV-1 --title "New"      # Update task
tv task start TV-1                     # Mark in_progress
tv task done TV-1                      # Mark done
tv task create "Subtask" --parent TV-1 # Create subtask
tv task block TV-2 --by TV-1           # Set dependency
```

### Projects

```bash
tv project list                        # List projects
tv project create "Name" --slug proj   # Create project
tv project get proj                    # Get details
```

### Server

```bash
tv serve                               # Start API server on :3000
tv serve -p 8080                       # Custom port
tv webui                               # Open WebUI in browser
```

## Configuration

Config file: `.taskvector/client_config.toml`

```toml
[api]
url = "http://localhost:3000"

[defaults]
projectSlug = "my-project"

[output]
format = "markdown"
```

## Status Values

| Status | Description |
|--------|-------------|
| `backlog` | Not yet prioritized |
| `todo` | Ready to work on |
| `in_progress` | Currently being worked |
| `done` | Completed |

## Priority Values

| Priority | Description |
|----------|-------------|
| `high` | Urgent |
| `medium` | Normal (default) |
| `low` | Can wait |
