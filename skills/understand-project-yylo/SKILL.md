---
name: understand-project-yylo
description: Inspect the current product architecture, dependencies, and validation loops before planning or implementing a requested change.
risk: safe
source: https://github.com/yylo-dev/yylo-skills/tree/main/skills/understand-project-yylo
---

# Understand the project

## Overview

A structured read-only investigation pass over a YYLO project before planning or implementing a change: read the instructions and sources of truth, trace only the needed dependency paths, and report the smallest useful validation loop.

## When to Use This Skill

- Use before planning or implementing a requested change in a YYLO project.
- Use when the user asks what the product does, how components connect, or how it is validated.
- Do not assume controller-private metadata exists in a product worktree, and do not create files during investigation.

## How to run the investigation

1. Read `AGENTS.md`/`CLAUDE.md`, repository status, relevant source, tests, and existing product documentation in the integration or assigned feature worktree.
2. Read related Kanban tasks and durable specs through the canonical metadata controller. Do not assume `.juno_task/plan.md` exists and do not materialize controller-private metadata in a product worktree.
3. Trace only the dependencies and runtime paths needed for the requested goal. Use bounded parallel investigation when independent questions justify it.
4. Report current behavior, sources of truth, affected components, risks, unknowns, and the smallest useful validation loop.
5. If the user requested planning, hand the findings to `plan-ledger-tasks-yylo`. If implementation was requested, work only in the task worktree returned by `yy task start TASK_ID`.
6. Write a durable spec only when requested or materially useful, and route it to a controller-admitted location. Do not update root instructions with transient status.

## Copy-paste example

```text
User: "Understand how billing webhooks are validated before we plan the retry change."
→ read AGENTS.md, trace webhook handler → validator → test suite,
  report affected components + smallest validation loop, hand off to planning.
```
