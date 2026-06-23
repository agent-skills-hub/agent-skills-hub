# iOS Testing Rules

## 1. Test layers

Use the project profile to identify actual schemes and targets. A healthy project normally separates:

- fast deterministic unit tests
- integration and persistence tests
- UI/system tests
- performance benchmarks

Do not collapse all categories into one daily test entry. A fast entry must remain fast, isolated, deterministic, and safe for frequent parallel execution.

## 2. Fast deterministic tests

A test belongs in the fast layer only when it:

- owns its dependencies
- does not launch the application
- does not use a real persistent store
- does not schedule real notifications or background work
- does not require network or uncontrolled file-system state
- does not depend on wall-clock delays
- does not depend on test order
- is safe under parallel execution, or explicitly isolated

Suitable examples:

- pure domain transitions
- validation
- formatting and mapping
- isolated ViewModel state
- repositories backed by fast in-memory test stores, when genuinely deterministic

## 3. Tests excluded from the fast layer

Keep these in integration, UI, or performance entries:

- UI tests and real app launch
- real persistence migration
- production store access
- real notification delivery
- `measure` and `measureAsync`
- arbitrary `Task.sleep`, timers, and long waits
- shared global state that cannot be isolated
- tests requiring order between cases

## 4. XCTest and Swift Testing

They may coexist. Do not migrate frameworks without a requirement.

Swift Testing may run tests in parallel. Shared state must be eliminated, isolated, or explicitly serialized. Framework choice does not remove the need for deterministic dependency ownership.

## 5. Isolation

Prefer:

- dependency injection
- in-memory stores
- temporary directories
- isolated `UserDefaults` suites
- mock or recording schedulers
- injectable clocks
- local fixtures

Avoid ordinary tests that depend on:

- `UserDefaults.standard`
- production singletons
- shared static caches
- real persistent containers
- real time
- mutable runtime sample data

If shared state is unavoidable, reset it before and after each test and document why isolation could not be achieved.

## 6. Time and async behavior

Do not add sleeps to wait for state propagation.

Prefer:

- injectable clock
- controlled scheduler
- manual time advancement
- async expectation tied to the actual state transition
- task cancellation or generation IDs for stale-result tests

A real delay is acceptable only for a genuine system boundary and must not enter the fast layer.

## 7. MainActor

Do not annotate an entire suite with `@MainActor` unless every case requires it.

Use main-actor isolation for code that truly owns UI state. Repository, parser, validator, migration, and pure domain tests should remain actor-independent unless the production contract requires otherwise.

## 8. Persistence and migration

- Use isolated containers.
- Test old-data migration with explicit fixture versions.
- Verify idempotency by running migration more than once.
- Verify partial and invalid data behavior.
- Never point tests at a user's or developer's production store.

## 9. UI tests

UI tests should cover critical end-to-end behavior that cannot be proven below the UI layer. They must not substitute for domain tests.

- Use accessibility identifiers.
- Prefer targeted waits over long default timeouts.
- Avoid repeated unnecessary launches.
- Never create order dependency between UI tests.
- Validate meaningful user outcomes, not only element existence.

## 10. Performance tests

Keep benchmark and measurement tests separate from daily fast feedback. Do not remove measurement code or weaken thresholds merely to make a performance test green.

## 11. Production changes for testing

Allowed when they improve architecture:

- dependency injection
- protocol or interface extraction
- clock/scheduler injection
- separating pure logic from UI or persistence
- controlled test/debug reset hooks

Not allowed:

- weakening business rules
- bypassing validation
- adding production branches that exist only to satisfy a test
- exposing private state without an architectural reason
- changing launch behavior only for UI tests

## 12. Regression-test standard

A bug fix must include a test that:

1. expresses the violated invariant
2. fails on the defective behavior
3. exercises the owning layer or real changed boundary
4. passes after the correction
5. remains deterministic

A test that only verifies a mock call is insufficient when the defect was in composition, persistence, time, or side-effect wiring.

## 13. Review after test changes

Report:

- changed test files
- test layer/scheme used from the project profile
- any new timeout, wait, sleep, serialization, or `@MainActor`
- production changes made for testability
- actual command and result
- untested system, device, permission, migration, or visual conditions

## 14. Prohibited fixes

Do not:

- delete assertions to make tests pass
- replace exact assertions with vague non-null checks
- increase timeout without finding the wait cause
- add sleeps to hide a race
- move slow tests into the fast entry
- make tests share mutable fixtures
- declare coverage from test count alone
