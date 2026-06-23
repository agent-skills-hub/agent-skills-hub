# Rendered UI Visual Review

## Evidence first

Preferred evidence order:

1. user-provided screenshot
2. simulator/device screenshot
3. Xcode Preview screenshot
4. code-only inspection

Code-only review must be labeled as visually unverified. Alignment, clipping, wrapping, density, and visual balance cannot be confirmed from source alone.

## Scan before repair

For the relevant surface:

1. inspect all supplied states/screens
2. record findings before changing implementation
3. group findings by shared root cause
4. prioritize
5. repair deterministic issues if requested
6. render again and compare

Do not require a separate user-confirmation round when the user already requested review-and-fix, unless a real product/design choice is unresolved.

## Core visual dimensions

### Hierarchy

- one clear primary information area
- primary action is obvious without competing buttons
- secondary and metadata content are visually subordinate
- no duplicated page/navigation/card titles

### Alignment

- consistent page edges
- stable card and row inner edges
- icon/text vertical alignment
- numeric value/unit alignment
- title, metadata, and trailing action columns
- no accidental edge changes between collapsed/editing states

### Spacing and density

- semantic rhythm from project tokens
- consistent sibling spacing
- enough separation between groups
- no oversized empty regions or compressed control clusters
- editing/deletion controls do not collapse primary content

### Typography and copy

- few clear type roles
- title length does not dominate or destabilize the screen
- text weight matches importance
- helper text is necessary and scannable
- English and long translations do not create deformed buttons, badges, or cards

### Icons

- consistent visual family
- scale and stroke fit adjacent text
- meaningful semantics
- icon-only controls have accessible labels and adequate hit targets
- decorative icon use does not create noise

### Cards, lists, and forms

- grouping is clear without excessive borders and shadows
- sibling rows share structure
- form labels, fields, units, validation, and actions remain aligned
- cards do not become nested containers without hierarchy value

### Actions

- one primary CTA per local task when possible
- destructive actions are clear but not over-emphasized
- secondary actions do not appear equal to primary actions
- long button labels are shortened before shrinking typography

### Adaptation

Check as applicable:

- smallest supported device
- long content
- empty and dense data
- editing and deletion states
- keyboard shown
- Dynamic Type
- English and longest supported language
- light/dark mode if supported
- safe area and orientation if supported

## Priority

- `P0`: unusable, clipped, overlapping, inaccessible, or wrong interaction hierarchy
- `P1`: visibly unprofessional hierarchy/alignment/structure or common long-text breakage
- `P2`: meaningful polish, density, icon, radius, shadow, or local spacing inconsistency
- `P3`: subjective preference or tiny non-systematic difference

## Finding format

```text
[Priority] Surface/state
Problem:
Why it matters:
Likely owner:
Recommended correction:
Acceptance evidence:
```

## Validation after repair

- compare the same state before/after
- confirm corrected issue
- check sibling states for regression
- check long copy and interaction controls
- do not claim “pixel perfect” without a reliable reference and rendered comparison
