---
name: ios-engineering
description: "End-to-end SwiftUI and iOS engineering skill for planning, implementation, review, debugging, UI validation, XCTest generation, app audits, and evidence-based verification. Use when working on native iOS product requirements, bug fixes, UI changes, release validation, or code review."
risk: safe
source: "https://github.com/greysonOuyang/ios-engineering-skill"
---

# iOS Engineering

## Overview

Use this skill for the full delivery lifecycle of an existing native iOS app:
planning product behavior, implementing scoped changes, reviewing code, debugging failures, validating rendered UI, adding focused tests, auditing the app, and verifying completion with evidence.

The skill is intentionally generic. Real module names, schemes, targets, design tokens, domain rules, and build commands must come from the active repository rather than being invented.

## When to Use This Skill

- Use when implementing or fixing SwiftUI or native iOS product behavior.
- Use when reviewing an iOS change for bugs, regressions, missing tests, or ownership problems.
- Use when debugging stale UI, persistence bugs, navigation issues, async races, or data consistency problems in an iOS app.
- Use when replicating, reviewing, or repairing rendered iOS UI based on screenshots, design specs, or simulator output.
- Use when generating focused XCTest coverage for a concrete iOS feature or contract.
- Use when auditing a release candidate or verifying whether an iOS requirement is actually complete.

## Do Not Use This Skill When

- The task is primarily web, backend, Android, or design-system work outside native iOS execution.
- The repository does not contain an iOS product and the task is not about SwiftUI or Apple-platform engineering.
- A more specific skill owns the problem, such as a dedicated design-system infrastructure skill or a non-iOS deployment workflow.

## Commands

Preferred entry points:

```text
/ios plan <requirement>
/ios implement <task>
/ios review <scope>
/ios debug <failure>
/ios ui <replicate|review|fix> <surface>
/ios test feature <feature>
/ios audit app
/ios verify <requirement>
```

Natural-language requests are valid equivalents when the user does not use slash commands.

## Step-by-Step Guide

### 1. Discover project truth before acting

- Read the repository's project profile, requirement artifact, domain rules, and architecture notes.
- Identify the authoritative state owner, persistence boundary, navigation owner, and validation path before changing behavior.
- Reuse existing components and flows before creating parallel structure.

### 2. Choose one primary mode

- `plan`: freeze product structure, UX, state flow, and acceptance criteria.
- `implement`: make the smallest safe production change under a defined scope.
- `review`: scan the relevant surface first, then report findings grouped by severity.
- `debug`: trace the earliest broken invariant, not just the visible symptom.
- `ui`: compare rendered output to design intent and preserve real functionality.
- `test`: add focused regression protection for one meaningful contract.
- `audit`: stage whole-app QA and cross-flow validation.
- `verify`: judge completion using code, test, runtime, and visual evidence.

### 3. Load only the references the task needs

- For architecture and requirement work, read `modes/plan.md` plus the linked product and interaction references.
- For production changes, read `modes/implement.md` and `references/core-engineering-rules.md`.
- For write-path, persistence, or synchronization issues, read `references/data-consistency.md` and `references/fallback-and-error-handling.md`.
- For code review or unexplained failures, read `references/bug-patterns.md`.
- For SwiftUI and rendered UI work, read `references/swiftui-and-ios-ui.md`, `references/ui-replication.md`, and `references/visual-quality-review.md` as needed.
- For testing, auditing, or verification, use the matching `modes/` file plus its related references.

### 4. Execute with explicit evidence

- Preserve already-correct behavior and non-goals.
- Validate with the repository's real build, test, runtime, and visual entry points whenever available.
- Separate confirmed facts, inferred risks, and unknowns.
- For write-path changes, inspect downstream effects such as lists, detail pages, derived stats, notifications, and stale references.

### 5. Report actionable output

- State the scope examined.
- State the decision, findings, or completed change.
- Include evidence and unresolved risks separately.
- Use templates under `templates/` when a durable report or execution prompt is needed.

## Examples

### Example 1: Implement a scoped bug fix

```text
/ios implement "Fix a SwiftUI sheet save flow where edited records appear on the detail screen but not in the parent list"
```

### Example 2: Review a branch for production risk

```text
/ios review current branch
```

### Example 3: Verify a feature before release

```text
/ios verify "Recurring reminder editing is complete for release candidate 1"
```

## Best Practices

- Do identify the real source of truth before changing stateful behavior.
- Do prefer `reuse -> extend -> refactor -> rebuild`.
- Do validate rendered UI with screenshots or simulator evidence before claiming visual fidelity.
- Do treat compile success as incomplete evidence for behavior changes.
- Don't invent product behavior, module names, token names, targets, or schemes.
- Don't patch ownership problems with fallback state, broad refreshes, sleeps, or duplicate caches.
- Don't weaken assertions or business rules just to make tests pass.

## Limitations

- This skill does not supply repository-specific build commands, module names, or product semantics.
- Some references include mixed English and Chinese guidance; teams expecting a fully English-only skill may want a localized follow-up pass.
- The skill is optimized for native iOS and SwiftUI workflows, not React Native or Flutter implementation.
