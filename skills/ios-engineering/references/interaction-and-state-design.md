# Interaction and State Design

## Freeze the interaction contract

Before implementation, define:

- entry point and owner
- default state
- primary and secondary actions
- completion result
- cancel, back, close, and dismissal
- loading and in-flight behavior
- invalid input and disabled state
- failure and retry
- empty and unavailable state
- confirmation and undo
- context retained after return

## State inventory

Use only relevant states, but do not design the happy path alone:

```text
initial
loading
ready
empty
editing
valid / invalid
submitting
success
partial success
error
permission denied
unavailable
stale / needs refresh
deleted / restored
```

## State ownership

For each state, distinguish:

- persisted business fact
- application workflow state
- view presentation state
- local editing draft
- derived display state

A view draft may temporarily diverge from the persisted owner, but the commit and cancellation rules must be explicit.

## Create/edit/delete loop

### Create

- where the new fact is committed
- which readers update
- whether the user remains in context or navigates
- failure and duplicate-submission behavior

### Edit

- draft initialization from current truth
- effective time or history semantics
- post-save invalidation and refresh
- stale edit conflict behavior when relevant

### Delete

- confirmation and irreversibility
- reference cleanup
- side-effect cancellation
- navigation after deletion
- empty-state transition
- undo/restore where supported

## External action loop

For notification, widget, watch, shortcut, background, or deep-link actions:

```text
external adapter
→ application command
→ domain transition
→ persistence
→ derived/side-effect refresh
→ external surface refresh
```

External surfaces must not invent independent business semantics.

## Context retention

Specify whether to retain:

- selected date
- filter and sort
- search query
- tab
- scroll position
- selected item
- expanded/collapsed state

Retention is a product choice, not an accidental side effect of navigation lifetime.
