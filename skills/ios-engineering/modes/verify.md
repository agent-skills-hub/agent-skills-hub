# Verify Mode

## Objective

Determine whether a requirement is complete using product, code, test, runtime, visual, and documentation evidence.

## Workflow

1. Read the authoritative requirement, frozen design, acceptance criteria, and explicit non-goals.
2. If no usable acceptance checklist exists, derive a compact one from current product truth before judging implementation.
3. Compare production behavior to every criterion.
4. Inspect the real write/read chain, state owner, persistence, side effects, and presentation.
5. Confirm the exact tests, target/scheme, execution command, and result.
6. Confirm affected builds succeed.
7. Obtain runtime or rendered evidence where tests cannot prove the behavior.
8. Check edge states, localization, accessibility, migration/old data, and critical regression paths as relevant.
9. Check that issue status and durable project documentation match reality.
10. Produce exactly one verdict:
   - `Pass`
   - `Conditional Pass`
   - `Fail`

## Evidence standard

A criterion is complete only when supported by one or more of:

- named production code path
- named deterministic test and execution result
- build result
- reproducible runtime result
- device/simulator validation
- screenshot or visual comparison
- migration or old-data validation
- updated authoritative requirement/review record

A checked box, implementation note, comment, mock-only test, or successful compile alone is insufficient.

## Per-item status

Use:

- `✅ Pass` — include evidence
- `❌ Fail` — include impact and blocker status
- `⚠️ Partial` — distinguish completed and missing parts
- `N/A` — explain why it does not apply

## Conditional pass

Use `Conditional Pass` only when:

- core behavior is correct
- remaining work is bounded and explicit
- the missing evidence or work is non-blocking for the claimed stage
- each condition has an owner and verification path

## Scope control

After a failed or conditional verification, define the next repair scope. Protect already-correct areas from repeated redesign and list the regressions that must be rerun.
