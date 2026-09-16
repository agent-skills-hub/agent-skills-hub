---
name: ralph-loop-yylo
description: Execute exactly one explicitly assigned YYLO Ledger task through the Ralph loop to a validated queued commit. Use only when the user explicitly requests ralph-loop-yylo.
risk: safe
source: https://github.com/yylo-dev/yylo-skills/tree/main/skills/ralph-loop-yylo
---

# Ralph loop: one assigned task to a queued commit

## Overview

The Ralph loop executes exactly one explicitly assigned YYLO Ledger task — admission, implementation, preflight, and queueing — and stops after the task is queued for review.

## When to Use This Skill

- Use only when the user explicitly requests ralph-loop-yylo for an assigned task.
- Do not select unrelated work, edit `tasks.md`, auto-tag releases, push, deploy, mutate production, or broaden scope because another issue is noticed.

## How it works

Read [references/implement.md](references/implement.md) completely and follow it.

Keep durable instructions concise and evidence-backed. Status belongs in the task response and runtime receipts, not `AGENTS.md`.

Controller checkpoints are best-effort local durability warnings after terminal metadata is durable. They never gate `yy pi`, `yy task`, `yy merge`, product commits, candidates, or releases.

## Copy-paste example

```bash
yy task start TASK_ID     # admit the task and get its worktree
yy task preflight TASK_ID # validate closure readiness before final validation
yy task finish TASK_ID    # record QUEUED with immutable closure
```
