---
description: Interactive feature planning - understand requirements, explore options, create junior-dev-ready tasks
---

# TaskVector Plan Skill

This skill guides you through an interactive planning workflow for breaking down features into well-structured tasks. YOU (the main chat instance) drive this process directly, enabling richer conversation with the user.

## When to Use

Use this skill when the user asks to:
- Build a new feature
- Add functionality
- Plan or design implementation
- Break down a user story into tasks

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: UNDERSTAND                                            │
│  Ask 2-3+ clarifying questions to fully grasp the request       │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: EXPLORE                                               │
│  Spawn explorer agents to investigate codebase patterns         │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: OPTIONS                                               │
│  Present 2-3 implementation approaches with different scopes    │
│  Compare pros/cons deeply - let user choose                     │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: DEEP DIVE                                             │
│  Explore again based on chosen approach                         │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 5: TASK BREAKDOWN                                        │
│  Create detailed, junior-dev-ready tasks in TaskVector          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Understand the Request

**Goal:** Deeply understand what the user wants before touching any code.

Ask **2-3 clarifying questions minimum** (more if needed). Use `AskQuestion` tool for structured choices, or conversational questions for open-ended topics.

### Questions to Consider

**Scope & Goals:**
- What problem does this solve? 
- What's the minimum viable version vs the ideal version?
- Are there existing patterns/features this should match?

**Technical Constraints:**
- Read your personal learnings knowledge base in bash script (it prints the db glossary you can read tasks' relavant cases from) 
Run this from the git repo root:
```bash 
  ./.taskvector/learnings/build_glossary.py
```
  
- Any specific tech/libraries to use or avoid?
- Performance requirements? Scale expectations?
- Backward compatibility needed?


### Example Question Flow

```
AskQuestion({
  title: "Understanding Your Feature Request",
  questions: [
    {
      id: "scope",
      prompt: "What scope are you targeting?",
      options: [
        { id: "mvp", label: "MVP - minimal working version, can refine later" },
        { id: "solid", label: "Solid - production-ready but not over-engineered" },
        { id: "comprehensive", label: "Comprehensive - full feature with all edge cases" }
      ]
    },
    {
      id: "bundler",
      prompt: "How should we bundle the changes?",
      options: [
        { id: "npm-bundler", label: "Npm bundler because it's the default" },
        { id: "pnpm-bundler", label: "Pnpm bundler because we use it" },
      ]
    }
  ]
})
```

**DO NOT proceed to Phase 2 until you have clear answers.**

---

## Phase 2: Explore the Codebase

**Goal:** Understand existing patterns before proposing solutions.

Spawn **explore** agents (can run in parallel) to investigate:

```
Task(
  description="Explore auth patterns",
  subagent_type="explore",
  prompt="Explore how authentication/authorization is currently handled in this codebase. Look for: middleware patterns, session/token handling, user models, permission checks. Report key files and patterns found.",
  readonly=true
)
```

### What to Explore

1. **Similar features** - How are comparable features implemented?
2. **Patterns in use** - What conventions does this codebase follow?
3. **Integration points** - Where would new code connect?
4. **Test patterns** - How are similar features tested?
5. **Consumers/dependencies** - If changing shared infrastructure (APIs, client libraries, utilities), who uses it?
   - **Principle:** Shared code has dependents - find them all before planning breaking changes
   - **Method:** Use `rg` (ripgrep) to search codebase-wide for references
   - **Examples:**
     - Find import statements: `rg "package_name" --files-with-matches`
     - Find in configs: `rg "package_name" **/package.json` or `**/Cargo.toml` or `**/requirements.txt`
     - Find function calls: `rg "functionName\(" --files-with-matches`
     - Find class usage: `rg "ClassName" --type=py` (or --type=js, --type=rust, etc.)
   - **CRITICAL:** All consumers must be updated for breaking changes

**Capture Learnings:** If you discover reusable patterns, codebase structure insights, or gotchas during exploration, write them to `.taskvector/learnings/raw/<topic>_<slug>.md` immediately. These help future tasks avoid repeating your research.

### Exploration Prompts by Feature Type

**API/Backend:**
```
"Explore API route patterns in this codebase. Find: route definitions, middleware usage, request validation, error handling, response formats. Note which files define routes and what patterns they follow."
```

**UI/Frontend:**
```
"Explore UI component patterns. Find: component structure, state management, styling approach, form handling, data fetching patterns. Identify reusable components and conventions."
```

**CLI:**
```
"Explore CLI command patterns. Find: command definition structure, option handling, output formatting, error handling. Note how existing commands are organized."
```

**Database:**
```
"Explore database patterns. Find: schema definitions, migrations, query patterns, ORM usage, transaction handling. Note naming conventions and relationships."
```

**Shared Infrastructure (APIs, Libraries, Clients):**
```
"Explore consumers of [package/module/function]. Find ALL usages across the codebase:
- Use ripgrep (rg) to search for imports, function calls, class usage
- Check dependency manifests (package.json, Cargo.toml, requirements.txt, etc.)
- Identify all affected components (CLI, WebUI, API, services, tests)
- Report: list of files that import/use this, plus dependency graph
This ensures breaking changes account for ALL consumers."
```

### Synthesize Findings

After exploration, summarize:
- Key patterns discovered
- Files that would be modified/referenced
- **All consumers affected** (if changing shared infrastructure)
- Conventions to follow
- Potential challenges or gotchas

**Write learnings:** Capture reusable insights to `.taskvector/learnings/raw/<topic>_<slug>.md`:
- Codebase structure (where things live)
- Patterns and conventions found
- Non-obvious gotchas or traps
- Consumer dependencies discovered

---

## Phase 3: Present Implementation Options

**Goal:** Give the user informed choices, not just one approach.

In any case but trivial present **2-3 distinct approaches** with different scope/effort tradeoffs:

### Option Template

```markdown
## Option A: [Name] (Shortcut / Quick Win)

**Approach:** [1-2 sentences]

**Scope:**
- [What's included]
- [What's NOT included]

**Effort:** ~X hours | Y tasks

**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Con 1 - be honest about limitations]
- [Con 2]

**Best if:** [When to choose this]

---

## Option B: [Name] (Balanced / Recommended)

**Approach:** [1-2 sentences]

**Scope:**
- [What's included]
- [What's improved over Option A]

**Effort:** ~X hours | Y tasks

**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Con 1]
- [Con 2]

**Best if:** [When to choose this]

---

## Option C: [Name] (Comprehensive / Full Refactor)

**Approach:** [1-2 sentences]

**Scope:**
- [Full feature set]
- [Additional improvements/refactoring]

**Effort:** ~X hours | Y tasks

**Pros:**
- [Pro 1]
- [Pro 2]

**Cons:**
- [Higher effort]
- [Other tradeoffs]

**Best if:** [When to choose this]
```

### Comparison Table

Always include a summary table:

```markdown
| Aspect | Option A | Option B | Option C |
|--------|----------|----------|----------|
| Effort | Low | Medium | High |
| Maintainability | Fair | Good | Excellent |
| Covers edge cases | Partial | Most | All |
| Future flexibility | Limited | Good | Full |
| Recommended for | Prototyping | Production | Long-term |
```

**Ask the user to choose** before proceeding.

---

## Phase 4: Deep Dive on Chosen Approach

**Goal:** Get detailed implementation specifics for the selected option.

After the user chooses, spawn another explore agent for targeted investigation:

```
Task(
  description="Deep dive [chosen approach]",
  subagent_type="explore",
  prompt="We're implementing [chosen approach]. Deep dive into: [specific areas]. Find exact file locations, function signatures, and patterns to follow. Report with file:line references.",
  readonly=true
)
```

### Gather Implementation Details

For each component of the chosen approach:
- Exact files to modify (with line numbers if possible)
- Functions/classes to add or change
- Imports and dependencies needed
- Error handling patterns to follow
- Test files and patterns to match

---

## Phase 5: Create Tasks

**Goal:** Create junior-dev-ready tasks in TaskVector.

### Task Principles

1. **Self-contained** - Each task has ALL context needed
2. **Prescriptive** - Tell them exactly what to do, don't assume great knowledge, just some generic coding skills
3. **Right-sized** - 1-4 hours each; decompose if larger
4. **Compliant to learnings system** - use the learnings glossary, read relevant cases from the glossary before creating the task
5. **Ordered** - Dependencies flow logically (schema → logic → API → CLI → UI)

### Task Structure

**Balance: Provide structure without full implementation.** Give the developer enough to understand the approach without writing all the code for them. Use explicit state transitions and behavioral specifications.

Use this template:

```markdown
# [Task Title]

## Design

**What triggers [behavior]:**
- [Condition 1] → [action]
- [Condition 2] → [action]

**What clears/reverses [behavior]:**
- [Condition] → [action]

**Other behaviors:**
- [Sorting/filtering/display changes]

## Implementation

### 1. [Layer/Component Name] (e.g., Schema Change)

[Brief description of what to do]

**File:** `path/to/file.ts`

```[lang]
// Key code snippet - field definition, interface, etc.
fieldName: type('column_name', { options }),
```

[Additional notes: indexes, constraints, etc.]

### 2. [Next Layer] (e.g., Migration)

[Instructions for this step]

**Commands:**
```bash
pnpm drizzle-kit generate
```

**Migration will:**
- [What the migration does]
- [Backfill logic if needed]

### 3. [Next Layer] (e.g., API Updates)

**File:** `path/to/routes.ts`

- `PATCH /endpoint` - [logic description]
  - If [condition] → [action]
  - If [other condition] → [action]
- `DELETE /endpoint` - [logic]
- `GET /endpoint` - [sorting/filtering logic]

### 4. [Additional Layers as needed]

[Continue pattern for CLI, WebUI, etc. if applicable]

## Edge Cases

- [Edge case 1]: [How to handle it]
- [Edge case 2]: [How to handle it]
- [Edge case 3]: [How to handle it]

## Tests

**Unit tests:** `path/to/__tests__/file.test.ts`
- [Test case 1]
- [Test case 2]

**Integration tests:** (if applicable)
- [Test scenario]

## Acceptance Criteria

- [ ] [Observable/testable outcome 1]
- [ ] [Observable/testable outcome 2]
- [ ] [Observable/testable outcome 3]
```

### Template Guidelines

**Design section:**
✅ State transitions (what triggers X, what clears X)
✅ Behavioral specifications
✅ Sorting/filtering changes

**Implementation section:**
✅ Numbered steps by layer (Schema → Migration → API → CLI → WebUI)
✅ Exact file paths
✅ Code snippets for key structures (5-15 lines max)
✅ Commands to run

**Edge Cases section:**
✅ Non-obvious scenarios
✅ Concurrent/bulk operation handling
✅ Data migration considerations

**Tests section:**
✅ Test file locations
✅ Key test cases to cover

**Acceptance Criteria:**
✅ Observable outcomes (not implementation details)
✅ Checkbox format for tracking

**What NOT to include:**
❌ Full line-by-line implementation
❌ Obvious/boilerplate code
❌ Complete functions - just show structure

### Example Task

```markdown
# Add resolvedAt Timestamp for Task Resolution Tracking

## Design

**What triggers resolvedAt to be set:**
- Status changed to "done" → set resolvedAt = now()
- Task archived → set resolvedAt = now()

**What clears resolvedAt:**
- Status changed from "done" to anything else → clear resolvedAt = null
- Task restored from archive → clear resolvedAt = null

**Sorting change:**
- Done/archived tasks sorted by resolvedAt DESC (most recently resolved first)
- Other tasks continue to sort by createdAt DESC

## Implementation

### 1. Schema Change

Add field to packages/db/src/schema.ts:

```ts
resolvedAt: integer('resolved_at', { mode: 'timestamp' }),
```

Add index for efficient sorting:

```ts
index('tasks_resolved_at_idx').on(table.resolvedAt),
```

### 2. Migration

Generate migration with `pnpm drizzle-kit generate`. The migration will:
- Add nullable resolved_at column
- Backfill existing done/archived tasks with resolved_at = updated_at
- Add index

### 3. API Updates

**File:** packages/api/src/routes/tasks.ts

- `PATCH /tasks/:id` - When status changes:
  - If new status is "done" and old status wasn't "done" → set resolvedAt = now()
  - If old status was "done" and new status isn't "done" → set resolvedAt = null
- `DELETE /tasks/:id` (archive) - Set resolvedAt = now() when archiving
- `POST /tasks/:id/restore` - Clear resolvedAt = null when restoring
- `GET /tasks` - Conditionally sort:
  - If filtering by status "done" or including archived → sort by resolvedAt DESC
  - Otherwise → sort by createdAt DESC

### 4. Client/Type Updates

- packages/db/src/schema.ts - Type already inferred
- packages/client/src/index.ts - Task type picks from API, auto-updated

### 5. No CLI/WebUI Changes Needed

The field is returned in API responses and sorting happens server-side.

## Edge Cases

- Task archived while status is "done": resolvedAt already set, don't overwrite
- Bulk operations: Each task gets its own resolvedAt timestamp
- Existing data: Backfill with updated_at as approximation

## Tests

**Unit tests:** packages/api/src/routes/__tests__/tasks.test.ts
- Status change to done sets resolvedAt
- Status change from done clears resolvedAt
- Archive sets resolvedAt
- Restore clears resolvedAt
- Done tasks sorted by resolvedAt DESC

## Acceptance Criteria

- [ ] resolvedAt set when task marked done
- [ ] resolvedAt set when task archived
- [ ] resolvedAt cleared when task reopened or restored
- [ ] Done tasks sorted by resolvedAt DESC
- [ ] Existing done/archived tasks backfilled
```

### Using Subtasks

Group related work under a parent task:

```bash
# Create parent task
tv task create "User authentication feature" -b "$(cat <<'EOF'
## Context
Add user authentication to the application.

## Subtasks
This is a parent task. Complete all subtasks below.

## Acceptance Criteria
- [ ] All subtasks complete
- [ ] End-to-end auth flow works
EOF
)"
# Returns TV-X

# Create subtasks
tv task create "Add User model and migration" --parent TV-X -b "[body]"
tv task create "Add auth middleware" --parent TV-X -b "[body]"
tv task create "Add login/register endpoints" --parent TV-X -b "[body]"
tv task create "Add auth CLI commands" --parent TV-X -b "[body]"
```

**When to use subtasks:**
- Related changes for one feature (same context, ship together)
- Natural dependency order within a feature

**When to use separate tasks:**
- Independent work items (different contexts)
- Can be shipped/tested separately

### Task Creation Commands

```bash
# Simple task
tv task create "Task title" -b "[body]"

# With priority
tv task create "Task title" -b "[body]" --priority high

# As subtask
tv task create "Subtask title" --parent TV-X -b "[body]"

# Multiline body with heredoc
tv task create "Task title" -b "$(cat <<'EOF'
## Design

**What triggers [behavior]:**
- [Condition] → [action]

## Implementation

### 1. [Layer]
...

## Edge Cases
- [case]: [handling]

## Tests
- [test case]

## Acceptance Criteria
- [ ] [criteria]
EOF
)"
```

### After Creating Tasks

1. List what was created: `tv task list -s backlog`
2. Report summary to user:
   ```
   Created X tasks for [chosen approach]:
   - TV-Y: [title] (parent)
     - TV-Y1: [subtask 1]
     - TV-Y2: [subtask 2]
   - TV-Z: [independent task]
   
   Recommended order: TV-Y → TV-Z
   ```
3. Ask user which task(s) to implement

---

## Learnings

After completing the workflow, output any discoveries and put them in .taskvector/learnings/raw/<topic_title>.md
example:
About code structure mostly and insight about how to find things in the codebase.

```
---
description: "db schemas location and migration #db #db_schema"
learned: "2026-01-24_18:05:00"
---
the schema files are in the packages/db/src drizzle folder and the migration files are in the packages/db/drizzle/ folder.

packages/db/src
```
---

## Workflow Summary

1. **UNDERSTAND** - Ask 2-3+ questions. Don't assume.
2. **EXPLORE** - Spawn explore agents. Learn patterns.
3. **OPTIONS** - Present 2-3 approaches. Compare deeply.
4. **DEEP DIVE** - Explore chosen approach in detail.
5. **CREATE TASKS** - Junior-dev-ready, prescriptive, self-contained.

**Key principle:** This is a conversation, not a one-shot task. Engage the user at each phase.
