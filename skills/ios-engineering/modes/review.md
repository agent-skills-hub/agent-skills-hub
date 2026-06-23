# Review Mode

## Objective

Find defects, missing contracts, and scope drift that a reasonable implementer may have overlooked. Review the runtime and product result, not only the diff description.

## Workflow

1. Resolve the intended behavior, review scope, and explicit non-goals.
2. Enumerate affected owners and user-visible surfaces.
3. Read the diff and enough surrounding code to reconstruct the real runtime path.
4. Trace every changed write to readers, derived outputs, side effects, and historical records.
5. Load relevant bug, consistency, fallback, testing, SwiftUI, and UI references.
6. Inspect tests and prove that they execute the changed production path.
7. For rendered UI changes, obtain screenshots or clearly label the review as code-only and visually unverified.
8. Scan the full relevant scope before beginning fixes.
9. Group findings by root cause and affected requirement, not by discovery order.
10. Classify scope and severity.
11. If `--fix` was requested, repair deterministic in-scope findings and rerun narrow validation.

## Finding standard

Every material finding must contain:

- confidence: `Confirmed | High-risk | Needs verification`
- severity: `P0 | P1 | P2 | P3`
- triggering condition
- observable or reachable failure
- code path or architectural cause
- user/product/data impact
- smallest root-cause correction
- missing regression evidence
- scope: `current | defer | historical | global-system`

Do not report style preference as a defect.

## Review dimensions

Apply automatically when relevant:

- product behavior versus requirement
- state ownership and refresh
- data/reference integrity
- historical snapshot correctness
- derived-data invalidation
- notification/cache/widget/watch/export drift
- async cancellation and stale result ordering
- repeated actions and idempotency
- navigation and presentation ownership
- permission and error visibility
- fallback and partial-data behavior
- persistence migration and old data
- localization and accessibility
- rendered layout and visual consistency
- test isolation and false confidence

## Fix policy

### Review only

Return the full relevant findings. Do not modify production code.

### Review and fix

1. Finish the relevant scan.
2. Preserve a stable issue list.
3. Fix confirmed, deterministic, current-scope issues.
4. Do not auto-fix product semantics or shared visual-language conflicts with multiple plausible outcomes.
5. Re-run tests and rendered validation.
6. Report what remains and why.

## Output

Order by severity, then include:

- already-correct areas that should not be changed again
- tests/builds/runtime/screenshots checked
- residual uncertainty
- next-round scope boundary
