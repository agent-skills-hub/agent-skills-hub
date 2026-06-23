# Debug Mode

## Objective

Explain an observed failure through a verified runtime chain, correct the earliest wrong assumption or state transition, and add regression protection that would have failed before the fix.

## Workflow

1. Restate the symptom with a reproducible trigger and actual result.
2. Define the violated product, data, or UI invariant.
3. Identify the reader that renders or returns the wrong result.
4. Trace backward through:
   - selector/query
   - derived state or cache
   - persisted or in-memory owner
   - state generation
   - original command/write
   - time, lifecycle, permission, account, and dependency boundaries
5. Check sibling consumers and external side effects for disagreement.
6. Reproduce with the narrowest deterministic test or harness available.
7. Inspect existing tests and explain the exact coverage gap.
8. Fix the earliest incorrect assumption, transition, ownership boundary, or invalidation rule.
9. Add a regression test at the layer that owns the invariant.
10. Run adjacent and cross-flow regressions proportional to risk.

## Do not stop at symptom patches

The following require proof and are usually insufficient alone:

- adding `reload()` to a view
- inserting a default entity or empty collection
- reconstructing missing history from current configuration
- increasing a timeout
- adding `Task.sleep`
- broadening a `catch` and continuing
- forcing view recreation with random identity
- duplicating data into a second store

## Root-cause classification

Classify the defect as one or more of:

- requirement or product contract gap
- state ownership
- write/read chain mismatch
- derived-data invalidation
- persistence/migration
- time/effective-date boundary
- async ordering/cancellation
- navigation/lifecycle
- external side-effect drift
- fallback/error handling
- test architecture gap
- rendered UI/state mismatch

## Output

Report:

- symptom and trigger
- expected invariant
- root cause with evidence
- why existing tests missed it
- change made or proposed
- regression test and validation command
- affected adjacent flows
- remaining uncertainty
