# Feature Test Mode

## Objective

Add focused, deterministic XCTest coverage for one feature, owner, state transition, service, persistence contract, or user interaction without redesigning the production architecture.

## Workflow

1. Read the requirement and locate the actual feature owner and existing tests.
2. Identify externally meaningful entry points:
   - store/view-model actions
   - domain/application commands
   - repository/service requests
   - decoding and mapping
   - persistence writes and queries
   - state transitions
   - interaction gating
   - cancellation, debounce, or stale-result behavior
3. Reuse existing dependency injection seams.
4. Add only the smallest seam needed for deterministic testing.
5. Choose the narrowest layer that proves the contract.
6. Cover success, failure, edge, transition, and cancellation behavior as applicable.
7. Run the narrowest real test command first.
8. Fix production defects at their owner; never weaken the test to fit wrong behavior.
9. Expand regression scope only when dependency or shared-contract risk requires it.

## Layer selection

Prefer:

1. pure domain/unit test
2. store/view-model/command test
3. repository or persistence integration test
4. composed SwiftUI harness
5. XCUITest

Use XCUITest only when the contract depends on actual hit-testing, rendered enabled state, native navigation containers, modal presentation/dismissal, focus, keyboard, or gestures.

## Required contract checks

When relevant, test:

- valid and invalid submission gating
- empty selection and batch-action gating
- duplicate-tap prevention
- loading/in-flight state
- explicit close/dismissal path
- native back navigation or expected gesture container
- destructive confirmation
- stale or out-of-order async completion
- cancellation cleanup
- empty, nil, partial, duplicate, large, and old persisted data
- clock, day boundary, time zone, effective date, and recurrence
- persistence isolation and relaunch-equivalent behavior

## Test integrity

- Do not call real network services.
- Do not share mutable global persistence or user defaults across tests without isolation.
- Inject time when business behavior depends on time.
- Avoid arbitrary sleeps; use controllable continuations, clocks, schedulers, or explicit synchronization.
- Use production composition where the invariant spans multiple real owners.
- Mark test-only data and launch behavior clearly.
- Do not chase an arbitrary coverage percentage at the expense of business contracts.

## Output

Report:

- contract protected
- selected layer and why
- files created or changed
- scenarios covered
- validation command and result
- remaining unautomated behavior
