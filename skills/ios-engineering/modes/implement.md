# Implement Mode

## Objective

Deliver a frozen requirement through the project's established architecture while preventing cross-page, persistence, side-effect, testing, and UI omissions.

## Workflow

1. Read the authoritative requirement, acceptance criteria, project profile, and explicit non-goals.
2. Inspect the current implementation, sibling flows, reusable components, services, repositories, and existing tests.
3. Write down the authoritative state owner and full read/write path.
4. List affected consumers and side effects before editing.
5. Decide failure behavior before adding any fallback.
6. Define the narrowest meaningful tests before or with production changes.
7. Implement the smallest coherent root-cause change.
8. Preserve behavior outside the requested surface.
9. Run the narrowest relevant build and tests, then expand only as risk requires.
10. Review the resulting diff against the frozen requirement and acceptance criteria.
11. Record evidence, unverified conditions, and any durable project-contract change.

## Pre-flight

Keep this short, but make it explicit:

```text
doc_impact: none | page | module | product
preserve: up to three already-correct behaviors
affected owners: named pages/services/stores or none
owner docs checked: requirement | domain | architecture | design | testing
```

## Affected-surface checks

Apply when relevant:

- persisted entities and migration
- derived, grouped, filtered, searched, or statistical outputs
- historical snapshots and audit records
- notification, background task, cache, widget, watch, Live Activity, export
- parent/child refresh and navigation state
- permissions, account, subscription, and feature gates
- localization, accessibility, Dynamic Type, keyboard, and small screens

## Prospective questions

Before changing a write path:

- Which owner contains the fact?
- Which readers depend on it?
- What must refresh, invalidate, cancel, or reschedule?
- Is the command idempotent?
- What happens on partial failure or relaunch?
- What happens across day, time zone, permission, account, and migration boundaries?

Before changing UI:

- Is there an existing token, primitive, or feature component?
- Does the structure survive editing, deletion, long text, localization, keyboard, and accessibility sizes?
- Who owns presentation and dismissal?
- Does the UI reflect actual persisted success rather than optimistic appearance only?

## Direct execution policy

When the user asked to implement or fix:

- do not stop for confirmation on deterministic, in-scope decisions
- do stop or clearly isolate a genuine unresolved product/design conflict
- do not produce a long design document unless it is required for future ownership
- prefer updating an existing owner document over creating a one-time recap

## Completion gate

Implementation is not complete until:

- current product behavior and non-goals are satisfied
- no alternate source of truth was introduced
- affected side effects and readers are synchronized
- deterministic regression tests protect the owned invariant
- build and required test entries pass
- rendered UI is checked when visual claims are made
- unverified device, OS, permission, migration, or visual conditions are stated
