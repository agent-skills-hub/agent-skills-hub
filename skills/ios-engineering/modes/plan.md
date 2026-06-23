# Plan Mode

## Objective

Turn an incomplete or solution-shaped request into one coherent, implementation-ready product decision with frozen scope, interaction, UI structure, data linkage, execution order, and acceptance criteria.

## Use when

- the user proposes a new control, page, entry, setting, or workflow
- multiple pages or modules are affected
- a requirement is likely to be misunderstood during implementation
- a UI problem may actually be a product or interaction problem
- the user asks for an implementation prompt, product review, UX review, or requirement design

## Workflow

### 1. Interpret intent

Identify:

- user scenario
- real user goal
- current friction or failure
- whether the requested solution is only a surface proposal
- explicit constraints and non-goals

Challenge or simplify the request when it adds unnecessary configuration, duplicate entry points, or an unnatural mental model.

### 2. Inspect existing product context

Before creating anything new, locate:

- current module and owner
- existing adjacent capability
- current flow entry and exit
- current data owner and consumers
- sibling pages and shared components
- existing product and design contracts

Prefer:

```text
reuse existing capability
→ extend current flow
→ refactor an unclear boundary
→ add minimal new capability
```

### 3. Make one product decision

Choose one primary direction:

- extend existing capability
- modify existing flow
- add a minimal new capability
- reshape or reject the proposed solution

Offer alternatives only when there is a material cost, ownership, safety, or interaction trade-off that cannot be resolved from current product truth.

### 4. Define impact surface

List only affected owners, including as relevant:

- pages and components
- entry, exit, navigation, and dismissal
- create, edit, delete, restore, and view paths
- state, data model, persistence, cache, notification, widget, watch, export
- list, detail, history, statistics, search, settings, and reports
- localization, accessibility, and visual states

Do not treat every possible page as affected. Explain why each listed surface is involved.

### 5. Freeze interaction

Specify:

- default state
- main path and secondary path
- trigger and discoverability
- save, cancel, back, dismissal, and context retention
- success, failure, loading, empty, disabled, confirmation, and undo behavior
- duplicate-action prevention
- what is deliberately not exposed to the user

### 6. Freeze UI structure

Define visual order rather than arbitrary pixel values:

- primary information
- secondary information
- primary and secondary actions
- persistent versus progressive content
- card/list/form structure
- title and copy constraints
- long text, small screen, keyboard, localization, and Dynamic Type behavior

Project tokens and components are selected during UI implementation; do not invent them here.

### 7. Freeze the data and page loop

For each operation, define which owner changes and which consumers update:

```text
create
edit
remove
restore
filter or date change
background or external action
```

Explicitly cover historical facts, derived data, notifications, exports, and external surfaces when they are affected.

### 8. Group execution

Group work by stable concern, not discovery order:

1. product boundary
2. interaction model
3. data and state chain
4. UI structure
5. visual details
6. edge states
7. regression and evidence

A phase may be implemented separately, but all affected dimensions must be known before the first edit.

### 9. Define acceptance

Every acceptance item must include:

- behavior or visual condition
- affected surface
- pass condition
- evidence type
- whether it blocks the claimed stage

## Output modes

### Default: compact decision

Return:

- real problem
- decision
- recommended behavior
- affected surfaces
- preserved behavior/non-goals
- implementation grouping
- acceptance checks

### Execution prompt

When the user asks for a prompt, use `templates/execution-prompt.md`. It must state scope, required investigation, frozen behavior, prohibited expansion, tests, evidence, and documentation updates.

### Full specification

Use only when explicitly requested or when the requirement spans multiple durable product contracts.

## Guardrails

- Do not write implementation code before product and interaction truth is stable.
- Do not make the user choose between equivalent options.
- Do not solve spatial pressure by adding a display-mode setting unless users genuinely need persistent mode control.
- Do not require an empty state CTA when the correct next step is unavailable or already visible elsewhere.
- Do not mix unrelated historical cleanup into the current requirement.
- Do not encode temporary implementation details as product truth.
