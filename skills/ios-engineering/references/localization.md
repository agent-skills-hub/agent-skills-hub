# Localization Engineering and Copy Review

## Architecture

- Store localized resources in the project's selected localization system.
- Use stable semantic keys; do not use source-language sentences as permanent keys.
- Keep display copy out of domain entities and persistence models.
- Localize at presentation boundaries or through a dedicated presentation formatter.
- Scan production code for hardcoded user-facing strings and prohibited source-language comments according to project policy.

## Copy design

- Translate intent, not word order.
- Prefer the shortest conventional action that remains unambiguous.
- Titles identify the destination or task; helper text explains only what the user still needs to know.
- Do not repeat the same meaning in title, subtitle, badge, and button.
- Resolve overlong copy in this order:
  1. remove redundant meaning
  2. use a standard platform term
  3. move explanation to helper text
  4. adjust layout only when the content genuinely requires it

## UI review

Test at least:

- narrow devices
- long product, record, or entity names
- English and the longest supported language
- buttons, segmented controls, tabs, badges, and navigation titles
- destructive and confirmation copy
- Dynamic Type where supported

Do not solve localization by shrinking text below the project's readable type scale.

## Migration

- Migrate in bounded batches.
- Run a scanner after each batch.
- Track missing and unused keys separately.
- Require human review for ambiguous domain terminology.
- Do not claim completion from string-count reduction alone; verify visible screens.
