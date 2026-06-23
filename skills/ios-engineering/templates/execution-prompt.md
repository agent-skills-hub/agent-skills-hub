# Execution Prompt

Implement the requirement below in the existing iOS project.

## Objective

[State the user-visible result and real problem being solved.]

## Required investigation before editing

- Read the authoritative requirement and project profile.
- Inspect the existing owner, similar flows, reusable components, and tests.
- Trace the current read/write chain and affected side effects.
- Confirm project build/test commands; do not invent schemes or targets.

## Frozen behavior

- [Default state and entry]
- [Primary and secondary flow]
- [Save/cancel/back/error behavior]
- [Create/edit/delete/page linkage]
- [UI structure or interaction constraints]

## Preserve

- [Already-correct behavior 1]
- [Already-correct behavior 2]
- [Already-correct behavior 3]

## Explicit non-goals

- [No unrelated feature]
- [No broad redesign/refactor]
- [No alternate write chain or fallback]

## Engineering requirements

- Use the existing source of truth and application/domain command path.
- Synchronize affected readers, derived outputs, notifications/background work, cache, widget/watch/export, and history as applicable.
- Keep time-dependent behavior injectable and deterministic.
- Reuse project tokens and components for UI changes.
- Cover long text, localization, Dynamic Type, keyboard, small screens, and error/empty states when relevant.

## Tests and evidence

- Add regression tests at the layer that owns the invariant.
- Run the narrowest test first, then required regression/build entries.
- Render and compare affected UI when visual behavior changes.
- Report exact commands, results, limitations, and unverified conditions.

## Documentation

Update only the authoritative requirement, durable contract, or review record whose truth changed. Do not create a standalone recap document unless it will be required by future implementation or maintenance.
