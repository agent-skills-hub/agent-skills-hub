# Requirement and UX Review

## Start from the user task

Do not accept “add a button”, “add a switch”, “add a mode”, or “open a sheet” as the final problem statement.

Resolve:

- user context
- desired outcome
- current friction
- frequency and urgency
- information needed to decide
- safest and shortest completion path

## Reduce decisions

Prefer:

- sensible defaults
- contextual actions
- progressive disclosure
- compact default state with expansion when needed
- direct feedback after completion
- placing information where the user acts

Avoid adding persistent settings or mode selectors merely to solve layout pressure or an unclear default.

## Complete the information loop

For a newly created or edited fact, determine whether it belongs in:

- creation and editing
- list summary
- detail
- history or calendar
- statistics/search/filter
- reminder or external action
- export/report
- deletion/undo/restore

Do not expose a field everywhere by default. Show it only where it supports the user's task, but never leave a broken or misleading loop.

## Information architecture

Identify:

- primary task and information
- secondary information
- persistent versus progressive content
- contextual versus global controls
- when details deserve a separate destination
- terminology consistency across siblings

## Interaction quality

Check:

- discoverability without clutter
- action cost and number of decisions
- disabled and unavailable behavior
- save/cancel/back/dismissal
- success/error feedback
- context retention
- confirmation and undo
- destructive-action safety
- duplicate-action prevention

## Mobile constraints

- preserve first-screen focus
- support one-handed use when practical
- avoid persistent controls that steal space from the main task
- handle keyboard, small screens, long text, large text, and dense data
- keep primary actions reachable without turning every page into a fixed bottom-bar layout

## Copy

- titles identify destination or task
- button labels describe the result
- helper copy explains only information not already expressed by structure or action
- remove duplicate title/subtitle/CTA meaning
- avoid internal implementation or policy language in user-facing text
- validate long translations separately

## Scope classification

Review findings belong to one of:

- current requirement — must resolve now
- related but non-blocking — schedule separately
- historical defect — separate owner
- global design/system issue — separate governance task
- subjective preference — do not treat as a defect
