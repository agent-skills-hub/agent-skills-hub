# Whole-App Audit Reference

## Discover the audit map

Build module order from:

- app entry and root navigation
- product documentation
- feature directories
- deep-link destinations
- data owners and shared services
- critical user journeys
- release scope

Do not copy a module order from another repository.

## Recommended ordering

1. entry/onboarding/account prerequisites
2. primary daily or core workflow
3. creation/editing/detail owners
4. history/search/statistics/reporting
5. reminders/background/external surfaces
6. profile/settings/entitlement/sync
7. shared design system and cross-cutting infrastructure
8. cross-flow and relaunch regression

Adapt to the actual product.

## Audit evidence matrix

For every module, track:

| Lens | Scope | Result | Evidence | Fix/decision |
|---|---|---|---|---|
| Functional | primary and destructive flows |  |  |  |
| Test | owned invariants and gaps |  |  |  |
| Interaction | gating/navigation/focus |  |  |  |
| Performance | visible responsiveness |  |  |  |
| Copy | semantics and redundancy |  |  |  |
| UI | rendered consistency |  |  |  |

## Critical cross-flow candidates

Select product-relevant flows such as:

- create → list → detail → edit → history/export
- scheduled/background action → persisted fact → all readers
- delete → reference cleanup → empty states → relaunch
- permission denial → recovery
- subscription/entitlement change → gated features
- migration/old data → current UI
- external action → app open/deep link

## Release blockers

Typical blockers:

- loss/corruption of user data
- primary action inaccessible or wrong
- duplicate irreversible action
- stale or contradictory records across owners
- crash or dead-end navigation
- broken migration or relaunch
- notification/background action writes incorrect facts
- required legal/privacy/permission flow missing

Visual polish alone is usually not a release blocker unless it makes critical content or actions unusable.
