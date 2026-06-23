# Core Engineering Rules

## Scope control

- Resolve the requested change and explicit non-goals before editing.
- Preserve existing behavior outside the requirement.
- Prefer a small coherent change over a broad architectural rewrite.
- Refactor only when required to remove the root cause or make the behavior testable.

## Sources of truth

- Each business fact must have one authoritative owner.
- Derived values may be cached but must be invalidatable and reproducible from the owner.
- Views, widgets, notifications, and exports are consumers, not alternate fact owners.
- Current mutable configuration must not rewrite historical facts.

## Dependency direction

- UI sends intent and renders state.
- Domain/application services enforce business transitions.
- Repositories persist and retrieve facts without inventing them.
- Adapters such as notification, widget, watch, and export use explicit application interfaces.

## Failure behavior

- Distinguish missing data, valid empty data, permission denial, query failure, and invariant violation.
- Expose errors near their cause with typed or structured information.
- Prefer safe stop and visible recovery over invented business facts.

## Time

- Inject clocks or `now` into business logic.
- Test day boundaries, time zones, recurring schedules, and effective dates explicitly.
- Do not use arbitrary sleeps as business synchronization.

## Change evidence

- State what changed and why.
- Name affected files and tests.
- Report actual commands and results.
- Identify untested device, OS, permission, migration, or visual conditions.
