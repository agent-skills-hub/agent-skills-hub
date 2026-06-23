# Recurring iOS Bug Patterns

## Purpose

This file records recurring bug patterns that Codex must actively check for.

Read this file when fixing bugs, reviewing a feature, investigating stale UI, handling data inconsistency, modifying navigation, changing async code, changing persistence, changing reminders, or preparing release validation.

General workflow belongs in `SKILL.md`. Domain-specific rules must be read from the active project profile and domain documentation.

---

## 1. Stale UI After Write

Pattern:

A source entity is created, edited, deleted, checked in, or undone, but other pages still show old data.

Common causes:

* multiple ViewModels cache the same entity list
* edit page saves then only calls `dismiss()`
* parent page does not observe source changes
* list was loaded only in `onAppear`
* derived data was not recalculated
* search result cache was not invalidated
* stats/calendar view uses stale snapshot

Required fix:

* identify the single source of truth
* make dependent pages observe it
* refresh or invalidate derived data
* avoid broad refresh or random identity
* add regression test or manual verification across pages

Check affected pages:

```text
current page
parent page
home/list
detail
search/filter/grouping
calendar
stats
empty state
notification/cache/widget, if present
```

---

## 2. Duplicate State Owners

Pattern:

The same business entity exists in multiple long-lived places.

Example:

```text
HomeViewModel.items
StatsViewModel.items
CalendarViewModel.items
```

Symptoms:

* one page updates but another page does not
* back navigation shows old values
* stats do not match list
* search result differs from detail page
* delete works visually but old references remain

Required fix:

* move truth to Repository / Store / database query
* keep ViewModel state as UI/draft state only
* derive page-specific display models from the truth
* remove hidden fallback arrays

Avoid:

* global mutable singleton patch
* broad reload on every navigation
* `.id(UUID())`
* duplicate caches with manual sync

---

## 3. Save And Dismiss Without Parent Sync

Pattern:

A sheet or pushed edit page saves data and dismisses, but the parent remains stale.

Common causes:

* parent owns stale array
* child saves to repository but parent never reloads
* `onAppear` does not rerun
* TabView kept parent alive
* callback missing
* database query not observed

Required fix:

Use one of:

* shared Store
* observable query
* repository publisher
* explicit `onSave`
* explicit parent refresh after successful save

Do not rely on `dismiss()` as data synchronization.

---

## 4. Delete Leaves Invalid References

Pattern:

An entity is deleted but detail, edit, search, stats, notifications, or navigation still reference its ID.

Symptoms:

* crash from force unwrap
* blank detail page
* deleted item appears in search
* notification still fires
* stats query fails
* back stack opens deleted object

Required fix:

* handle missing entity safely
* remove or invalidate references
* cancel side effects
* update search/cache/stats
* return safely or show unavailable state
* avoid force unwrap of database result

---

## 5. Historical Records Mutate After Entity Edit

Pattern:

A historical log displays latest entity fields instead of the values at event time.

Symptoms:

* old record name/amount/category changes after editing current entity
* stats for past events become misleading
* exported history changes unexpectedly

Required fix:

* store snapshot fields in historical record
* display history from snapshot
* keep related entity ID only as optional reference
* do not rewrite history unless user edits that history

---

## 6. Derived Data Not Recomputed

Pattern:

The app updates the base entity but not derived displays.

Affected derived data:

* sorted list
* grouped list
* filtered list
* search result
* calendar model
* stats
* trend
* badge
* next item
* completion state
* inventory/balance
* empty state

Symptoms:

* item remains in wrong group
* sorting order wrong after edit
* old search result appears
* stats unchanged
* badge count wrong
* empty state incorrect

Required fix:

* derive display data from current source
* invalidate cache/index when source changes
* avoid manual patch of only visible row when change affects derived views

---

## 7. Notification Or Side Effect Drift

Pattern:

Business data changes but external side effects are not updated.

Examples:

* deleted item still has scheduled notification
* time change creates duplicate notifications
* notification content uses old name
* disabled item still triggers side effect
* permission denied state ignored
* widget/Live Activity shows stale data

Required fix:

* use stable side-effect identifiers
* cancel obsolete side effects
* recreate from latest persisted data
* handle permission denied and operation failure
* avoid duplicate scheduling
* clean up after delete/disable

---

## 8. MainActor Overuse

Pattern:

Entire ViewModel / Store / Repository / Service is marked `@MainActor`.

Symptoms:

* input lag
* keyboard delay
* scrolling jank
* search blocks UI
* database or file work runs on main thread
* large transforms block animations

Required fix:

* keep heavy work outside MainActor
* only publish final UI state on MainActor
* split methods that both compute and update UI
* avoid main-thread database/file/search work

---

## 9. Heavy Work In SwiftUI Body

Pattern:

`body` or frequently-read computed properties perform expensive work.

Examples:

* database fetch
* filter/sort/group large array
* fuzzy search
* JSON decode
* image processing
* complex date calculation
* formatter creation
* object graph rebuild
* side effects

Symptoms:

* typing is slow
* scroll is janky
* CPU spikes
* repeated rendering
* UI freezes on navigation

Required fix:

* move heavy work to ViewModel/Service
* precompute display models
* cache carefully with invalidation
* debounce high-frequency updates
* keep `body` as pure display

---

## 10. onAppear / onChange Reload Loop

Pattern:

`.onAppear`, `.task`, `.onChange`, or `.onReceive` triggers reload repeatedly.

Symptoms:

* repeated database fetch
* repeated network/local query
* repeated notification scheduling
* flickering loading state
* battery/CPU spike
* state toggles cause loop

Required fix:

* use `.task(id:)` when lifecycle-bound
* guard repeated loads
* cancel stale tasks
* make triggers specific
* avoid changing the same state that triggers the observer unless intentional

---

## 11. Search Per Keystroke Is Too Expensive

Pattern:

Every character triggers database fetch, large array scan, fuzzy search, sorting, or grouping.

Symptoms:

* typing lag
* keyboard stutter
* high CPU
* outdated results flash
* wrong result order from stale tasks

Required fix:

* debounce
* cancel stale searches
* use background work
* use SQLite / FTS / precomputed index for large local datasets
* publish only final results to MainActor
* avoid SwiftData high-frequency fetch for large catalogs

---

## 12. Async Result Overwrites Newer State

Pattern:

An old async task finishes after a newer task and overwrites current UI.

Symptoms:

* search result jumps backward
* date tab shows wrong data
* old detail load replaces newer detail
* save result applies to already-deleted entity
* navigation closes unexpectedly

Required fix:

* cancel old tasks
* track request identity
* use `.task(id:)`
* verify entity still exists before applying result
* merge result only if it matches latest user intent

---

## 13. Duplicate Submit / Repeated Tap

Pattern:

Fast repeated taps create duplicate writes or side effects.

Affected flows:

* save
* delete
* submit
* state transition
* undo
* import
* export
* notification scheduling
* payment-like operation

Required fix:

* guard with submitting state
* disable button while submitting
* make operation idempotent where possible
* avoid duplicate records
* avoid duplicate side effects
* handle retry after failure

---

## 14. Error Handling Only Prints

Pattern:

Code catches an error but only uses `print(error)`, `try?`, or empty `catch`.

Symptoms:

* user thinks save succeeded but data is lost
* loading spinner never stops
* notification silently not scheduled
* import fails with no explanation
* UI enters inconsistent state

Required fix:

* show user-visible error
* stop loading state
* allow retry when appropriate
* do not show final success before persistence/side effects succeed
* use structured logging without sensitive data

---

## 15. Navigation Owner Confusion

Pattern:

A child page directly mutates ancestor navigation state, tab state, or route path.

Symptoms:

* save jumps to unexpected page
* back button returns wrong place
* sheet closes too much
* route state duplicated
* deep link breaks normal navigation
* one flow fix breaks another flow

Required fix:

* presenting layer owns route
* child reports intent/result
* source page decides next route
* save/close normally ends current presentation only
* exceptions must be owned by the source flow

Avoid:

* global route mutation
* broad path reset
* tab switch as save side effect
* second navigation state to patch one bug

---

## 16. Keyboard Dismissal Breaks Interaction

Pattern:

A global transparent layer or UIKit gesture is used to dismiss keyboard.

Symptoms:

* buttons stop responding
* scroll gestures conflict
* tap targets broken
* keyboard state causes reload
* layout jumps

Required fix:

* use `@FocusState`
* dismiss by setting focus to `nil`
* use `.scrollDismissesKeyboard(...)`
* add toolbar done button where appropriate
* do not add full-screen invisible tap layers

---

## 17. Identity Bugs In Lists

Pattern:

List identity uses indices, unstable `\.self`, computed UUID, or random `.id()`.

Symptoms:

* wrong row updates
* animation glitches
* row state leaks
* deletion removes wrong row
* stale UI appears fixed but state is actually reset
* scroll position resets unexpectedly

Required fix:

* use stable business IDs
* avoid random identity
* avoid `.id(UUID())` refresh hacks
* ensure ID survives edits and reloads

---

## 18. Persistence Model Change Breaks Old Data

Pattern:

Model fields, enums, relationships, or table structure change without migration.

Symptoms:

* app crashes on launch after update
* old records disappear
* decode fails
* enum unknown case crashes
* relationship missing causes force unwrap crash

Required fix:

* add default values
* add migration/compatibility
* preserve old fields when needed
* handle unknown enum cases
* avoid destructive migration unless approved

---

## 19. Permission Happy Path Only

Pattern:

Feature assumes permission is granted.

Affected areas:

* notification
* camera
* photo library
* contacts
* location
* health data
* file access

Symptoms:

* button does nothing
* feature silently fails
* user cannot recover
* repeated permission prompts not possible
* settings redirect missing

Required fix:

* check authorization state
* handle denied/restricted/notDetermined
* provide user-visible explanation
* route to Settings when appropriate
* handle operation failure after permission is granted

---

## 20. Final Bugfix Checklist

Before finishing a bug fix, check only relevant items:

```text
Root cause identified:
Single source of truth checked:
Dependent screens checked:
Derived data checked:
Invalid references checked:
Side effects checked:
Async cancellation checked:
MainActor/body performance checked:
Duplicate submit checked:
User-visible error checked:
Navigation owner checked:
Regression test or manual verification:
```

Use `N/A` where irrelevant.
