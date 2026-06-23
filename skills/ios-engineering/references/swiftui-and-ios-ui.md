# SwiftUI and iOS UI Review

## State ownership

- A view must not own a second mutable copy of a business fact without an explicit draft boundary.
- Parent and presented views must define how successful writes propagate.
- Do not rely only on `onAppear` for post-write synchronization.
- Avoid reload loops caused by `onAppear`, `onChange`, task identity, or mutually updating bindings.

## Rendering performance

- Keep persistence queries, parsing, formatting of large collections, and expensive sorting out of `body`.
- Make asynchronous results cancellable or versioned so stale work cannot replace newer state.
- Debounce expensive search while allowing cheap local filtering where appropriate.

## Navigation and presentation

- Define the owner of push, sheet, full-screen cover, dismissal, and deep-link routing.
- Do not assume a pushed-page gesture works for a modal route.
- Dismiss only after a successful committed action unless the product explicitly chooses optimistic dismissal.
- Ensure modal and navigation actions do not compete for the same gesture or state flag.

## Layout

- Preserve safe areas unless a deliberate background treatment requires ignoring them.
- Avoid fixed heights for text containers that must support localization or Dynamic Type.
- Keep visible control size and touch target separate when needed; interactive targets should meet platform expectations.
- Validate keyboard behavior for scrollable forms and bottom actions.
- Validate small screens, long content, empty states, editing controls, and destructive controls.

## Components

- Reuse the project's existing primitives before adding new ones.
- Create feature components for business-specific composition.
- Promote a component to the design system only after it is generic, configurable, and reused.
- Do not bind global primitives to domain models.

## Accessibility

- Use semantic labels, traits, and identifiers.
- Do not communicate status only by color.
- Preserve reading order and sufficient contrast.
- Verify icon-only actions and combined rows have meaningful accessibility names.
