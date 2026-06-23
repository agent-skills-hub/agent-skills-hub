# Whole-App Audit Mode

## Objective

Run staged, evidence-based QA across the complete iOS product after major feature work or before release, while separating deterministic fixes from product/design decisions.

## Workflow

1. Discover the app's real modules, entry points, test targets, build schemes, and critical cross-flow journeys.
2. Build a module-based audit todo from repository truth; do not assume a fixed module list.
3. Audit one module at a time across:
   - functional chain
   - automated coverage
   - interaction contracts
   - responsiveness and obvious performance risk
   - copy and localization
   - UI consistency and rendered stability
4. Classify every result:
   - `pass`
   - `simple-fix`
   - `issue-for-decision`
   - `defer-nonblocking`
5. When repair was requested, fix only deterministic issues with clear ownership and expected behavior.
6. Re-run the narrowest module regression before moving on.
7. After modules pass, run critical cross-flow regression and release-target build/test entries.

## Per-module coverage

### Functional

- entry is reachable from the correct owner
- main flow closes correctly
- create/edit/delete/save/refresh/back/return paths work as applicable
- cross-module writeback reaches the correct owner
- permissions and unavailable states are handled

### Automated tests

- success, failure, edge, transition, cancellation, and interaction contracts
- real changed production path rather than mock-only behavior
- missing integration or UI coverage only where necessary

### Interaction

- actionable controls are reachable and correctly gated
- loading blocks duplicate actions
- dismiss/back paths exist
- keyboard/focus behavior is coherent
- context is preserved after navigation where required

### Performance

- immediate tap feedback
- no obvious main-thread work in typing, scrolling, navigation, or sheet presentation
- long lists and calculated surfaces remain responsive

### Copy

Classify helper text as:

- `delete`
- `rewrite-to-structure`
- `keep-as-exception`

Check action semantics, duplicate wording, long translations, internal-rule language, and unnecessary explanations.

### UI consistency

- shared edges, row structure, title placement, icon/text alignment, action hierarchy, radius, spacing, and safe area
- long content, localization, Dynamic Type, keyboard, and small-screen stability

## Simple-fix rule

Fix directly only when:

- expected behavior is already clear from product, domain, design, or code ownership
- no material product/design trade-off exists
- the change is local or clearly owned
- the fix does not redefine a shared semantic family

## Issue-for-decision rule

Create a decision item when:

- product truths conflict
- sibling flows disagree without an owner
- shared component semantics have several plausible outcomes
- copy meaning requires product judgment
- a fix changes module ownership or a stable design language

Each item includes:

```text
flow entrypoint
affected owners
problem
why not auto-fix
recommended option
acceptance check
```

## Final output

For each module:

- scope tested
- passes
- fixes
- decisions needed
- evidence and rerun command

Final summary:

- modules completed
- critical flows completed
- automated coverage added
- remaining issues by severity
- release blockers
- recommended final validation
