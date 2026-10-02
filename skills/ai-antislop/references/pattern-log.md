# Pattern log

Append-only log of **actual** slop catches — not near-misses, not
hypotheticals. Purpose: spot recurring patterns and turn them into
sharper rules instead of relying on "try to remember better."

This file follows the Laws too: log real instances only, don't inflate
counts, don't log something to justify a change someone already wants.

## Format

One line per entry:

```
YYYY-MM-DD | Law # | domain | what happened (one line)
```

**Example (illustrative only — not real log data):**
```
2026-09-20 | Law 1 | code | added a retry helper nobody asked for
  while fixing a bug — caught before sending, converted to an offer.
2026-09-24 | Law 9 | research | relayed a search result as confirmed
  fact instead of "menurut sumber X" — user caught it, corrected.
```

## Review cadence

- **Every ~10 entries**, or whenever asked ("gimana pola slop-nya
  sejauh ini?"), scan for repeats.
- **Same Law + same domain, 3+ times** → the rule isn't specific enough.
  Propose an actual addition/tightening to that Law or to
  `references/domains.md`, don't just note "be more careful."
- When a proposed change is accepted, record it in the Changelog
  section of `references/scenarios.md`, and note here which entries it
  resolved.

## Consolidation (keep this file from growing forever)

- Keep the most recent ~20 raw entries.
- Once older entries roll off, compress them into a short dated summary
  block here instead of deleting them silently:

```
## Summary: 2026-06 to 2026-09 (18 entries, pre-consolidation)
- Law 1 / code: 5x — resolved by adding baseline-competence distinction (v1.3)
- Law 3 / research: 4x — resolved by adding confidence-label vocabulary (v1.3)
- Law 9 / research: 3x — still recurring, watch
```

---

## Log

2026-09-27 | Law 4 | web | typo'd class attribute (`veel`) in shipped
  deliverable HTML — caught in proofread before verify, fixed.

*(tambah entri baru di atas baris ini)*