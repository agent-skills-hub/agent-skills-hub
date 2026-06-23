# Product Architecture for Existing iOS Apps

## Existing product first

Assume the product already has owners, routes, data models, and user habits. Confirm them before adding structure.

Default evolution order:

```text
reuse
→ extend
→ clarify/refactor boundary
→ add minimal new capability
→ rebuild only when the current structure cannot support the contract
```

## Product decision categories

Choose one:

- **Extend**: current owner and user path are correct; add behavior there.
- **Modify**: capability exists, but flow, information, or state handling is wrong.
- **Introduce**: a genuinely new owner or user task is required.
- **Reshape/reject**: the requested control is not the best solution to the user problem.

## Ownership questions

- Which module owns the user's task?
- Where should users naturally look for it?
- Is the proposed entry point duplicating another owner?
- Does the capability belong to a page, shared component, application service, or domain service?
- Is a new module carrying a real independent responsibility, or only avoiding changes to an existing owner?
- Will this create competing navigation, state, or terminology?

## Duplication risks

Treat these as product architecture defects unless explicitly intended:

- the same action on several unrelated pages with different semantics
- parallel settings for one behavior
- two independent histories or dashboards for one fact
- a new “center” that only re-lists existing destinations
- a secondary surface that writes business facts through a different chain
- multiple names for the same domain concept

## Impact check

Evaluate only relevant areas:

- product/user flow
- state and data model
- notification/background/external surfaces
- module ownership and coupling
- navigation/deep-linking
- history/report/export semantics
- entitlement or permission boundary

Do not produce empty architecture sections.

## Decision quality

A good product architecture decision:

- solves the actual user goal
- uses one clear owner
- minimizes new user decisions
- remains implementable in the current codebase
- preserves established behavior unless the requirement changes it
- has a clear migration path for existing data and users
- avoids speculative future layers
