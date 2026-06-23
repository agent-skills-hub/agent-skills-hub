# UI Replication Methodology

## Goal

Balance visual fidelity, maintainable architecture, design-token reuse, system behavior, accessibility, and localization.

## 1. Analyze before coding

For every screen or component, identify:

- visual hierarchy and grouping
- layout structure and alignment
- typography roles rather than only point sizes
- spacing rhythm
- semantic colors
- border, radius, shadow, and material
- interactive affordances and states
- safe-area, keyboard, scrolling, and navigation behavior

Create a measurement record when the screen is substantial. A measurement record should describe design intent and token mapping, not merely a list of raw pixels.

## 2. Token mapping

1. Read the project's current token definitions.
2. Map design values to the closest semantic token.
3. Prefer semantic meaning over exact numeric coincidence.
4. Extend the token system only when the value is repeated, meaningful, and cannot be represented by an existing token.
5. Record the reason for any new token.

Do not assume token names from another project; resolve the active repository's semantic token API first.

## 3. Component decision

Use this decision path:

```text
Existing primitive?
├─ yes: reuse it
└─ no
   ├─ composition of existing primitives works: create a feature component
   └─ no
      ├─ generic and reused across features: add a primitive
      └─ feature-specific: keep it in the feature
```

A design-system primitive must not contain business behavior or depend on a domain entity.

## 4. Fidelity priorities

P0:

- hierarchy
- major alignment
- screen and card structure
- typography role and weight
- primary color and contrast
- interaction semantics

P1:

- internal spacing
- radius
- border and shadow
- icon scale and alignment
- density

P2:

- tiny spacing differences that do not change rhythm
- nonessential animation tuning

## 5. System adaptation

Preserve the design intent while correcting platform-critical issues:

- usable touch targets
- safe-area behavior
- keyboard avoidance
- Dynamic Type and localization resilience
- semantic controls and accessibility

Do not redesign brand expression or product interaction merely because a platform default differs.

## 6. Validation

Validate through previews, simulator/device screenshots, and overlay or side-by-side comparison when available.

Check applicable states:

- default
- loading
- empty
- error
- disabled
- selected
- editing
- deletion
- long text
- English and another supported language
- larger text
- smallest supported screen
- keyboard shown

Report differences by impact, not by arbitrary pixel count alone.

## 7. Review questions

- Did editing controls change the outer card width or alignment?
- Do delete actions compress or wrap primary content?
- Are labels based on reliable domain meaning, or guessed from time/order?
- Can dose, unit, time, title, and actions coexist with localization?
- Are title lengths and line breaks intentional?
- Are secondary controls visually subordinate without becoming undiscoverable?
- Did the implementation add a feature that was not in the requirement?
