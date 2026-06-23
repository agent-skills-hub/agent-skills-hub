# UI Mode

## Objective

Replicate, review, or repair a rendered iOS interface while preserving product interaction, project design-system consistency, accessibility, localization, and maintainable component boundaries.

## Submodes

- `replicate`: implement from a design, screenshot, or specification.
- `review`: inspect actual rendered quality and compare it with product/design intent.
- `fix`: correct known visual or interaction defects with minimum disruption.

## Shared workflow

1. Read the project UI profile, design system, existing tokens, primitives, and sibling screens.
2. Identify all required states and interactions, not only the default screenshot.
3. Obtain rendered evidence where possible: user screenshot, simulator screenshot, preview, or test harness.
4. Inventory reusable primitives and feature components.
5. Describe hierarchy, alignment, typography, spacing, color, radius, border, shadow, icon proportion, safe area, and action placement.
6. Map design intent to semantic project tokens.
7. Decide `reuse | compose | extend | create` for each component.
8. Implement or review default, empty, loading, error, editing, deletion, long text, localization, Dynamic Type, keyboard, and small-screen states as applicable.
9. Compare rendered results after changes.
10. Recheck interaction, navigation, accessibility, and code boundaries after visual convergence.

## Replicate

- Treat the supplied design as visual intent, not permission to remove real functionality.
- Measure and map before coding.
- Preserve current runtime behavior unless the requirement explicitly changes it.
- Surface true design/function conflicts instead of silently choosing.
- Use project token semantics; do not import example token names from this skill.

## Review

Before fixing:

1. identify the complete affected visual surface
2. inspect screenshots across relevant states and languages
3. record all meaningful findings
4. group by hierarchy, alignment, density, typography, iconography, action placement, copy, and adaptation
5. assign priority

If screenshots are unavailable, state: `基于代码推断，未经渲染截图验证`.

## Fix

- Preserve the recorded issue list.
- Fix deterministic current-scope defects in priority order.
- Re-render after each coherent fix batch.
- Confirm the fix did not create wrapping, clipping, safe-area, keyboard, or sibling-component regressions.

## Visual priorities

- `P0`: broken hierarchy, layout structure, clipping/overlap, inaccessible action, wrong interaction semantics.
- `P1`: major alignment, type role, action hierarchy, inconsistent component structure, long-copy breakage.
- `P2`: internal spacing, radius, border, shadow, icon proportion, visual density.
- `P3`: subjective polish or very small non-systematic differences.

## Hard rules

- Do not claim visual fidelity without rendered evidence.
- Do not use fixed frames that break localization or accessibility unless the component contract explicitly requires them.
- Do not create a global primitive for a one-off business pattern.
- Do not add more helper copy to compensate for weak hierarchy or affordance without first fixing the structure.
- Do not repeat the same meaning in navigation title, page title, subtitle, and CTA.
- Do not introduce exact numeric values when the project requires semantic token reuse and the design does not define a unique exception.
