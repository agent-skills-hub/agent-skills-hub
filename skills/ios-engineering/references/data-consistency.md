# iOS Swift Guardrails: Data Consistency and State Synchronization

## Objective

When using Codex or another AI agent to build iOS Swift features, avoid stopping at the current screen while missing cross-screen, cross-owner, or cross-lifecycle data consistency issues.

Any write-oriented task involving create, edit, delete, status changes, undo, scheduling, balance or inventory updates, import, or migration must be checked against these rules.

---

## 1. Single Source of Truth

Core business data must come from one authoritative source, such as:

* Repository
* Store
* SwiftData / SQLite
* App-level observable state

Do not let multiple pages or multiple view models maintain long-lived independent copies of the same core data.

View models may store UI state, but they must not become the long-term source of truth.

Common bad tendency:

```swift
HomeViewModel.items
CalendarViewModel.items
StatsViewModel.items
```

Preferred direction:

```text
Database / Repository / Store
        ↓
HomeViewModel
CalendarViewModel
StatsViewModel
DetailViewModel
```

---

## 2. Check the Full Write Impact Chain

After any create, edit, or delete operation, verify that every surface referencing the entity updates correctly.

At minimum, check:

* home screen
* list screens
* detail screens
* edit screens
* search
* filter / grouping
* calendar
* statistics
* notifications
* widgets / Live Activities, if present
* cache
* empty states
* error states

Saving successfully and dismissing the current page is not enough to call the feature complete.

---

## 3. Derived Data Must Not Become the Long-Term Truth

The following are derived data and should not be stored as the long-term source of truth:

* sorted results
* grouped results
* filtered results
* search results
* "today" status
* next reminder
* statistics
* trend data
* inventory remaining
* completion rate
* calendar presentation data

When the underlying fields change, derived data must be recomputed from the latest authoritative source.

---

## 4. Deletion Must Handle Invalid References

When deleting a core entity, handle the case where other screens or data structures still hold the old identifier.

After deletion, check:

* whether the detail screen exits safely or shows an unavailable state
* whether the edit screen prevents saving an invalid object
* whether search results remove the deleted item
* whether stats skip invalid references
* whether related reminders are cancelled
* whether related records are retained, archived, nulled out, or cascaded intentionally
* whether the UI avoids crashes from force-unwrapping missing data

Never assume a database lookup result must exist.

---

## 5. Historical Records Must Preserve Necessary Snapshots

Historical data must not rely entirely on the current mutable entity fields for display.

For example, execution logs, status records, order history, and audit events should preserve the necessary snapshot fields from the time of the event, so future edits do not rewrite history.

At minimum, consider storing:

* name at the time
* dosage / quantity / unit at the time
* category at the time
* planned time at the time
* actual status at the time
* note at the time

---

## 6. Notifications Must Stay in Sync with Business Data

If reminder-related fields change, the system notifications must be updated as well.

When a reminder changes, the flow must:

1. cancel the old notifications for that entity
2. recreate notifications from the latest data
3. use stable, traceable notification identifiers
4. handle notification permissions being disabled
5. avoid duplicate notifications
6. handle delete, disable, time changes, name changes, and similar scenarios

Do not only add new notifications without removing the old ones.

---

## 7. Page Refresh Cannot Depend Only on `onAppear`

`onAppear` must not be the only refresh mechanism for core data.

Account for cases such as:

* `TabView` pages not being destroyed frequently
* the parent page not refreshing automatically after a sheet closes
* navigation back returning to a screen whose data has changed
* app foregrounding after stale background time
* data being modified on another screen

Core screens should react to change through a store, query, observable state, publisher, callback, or an explicit refresh path.

---

## 8. Sheet / Navigation Writes Must Notify Upstream Owners

When a create or edit page is presented via `sheet`, `fullScreenCover`, or `NavigationLink`, saving successfully must ensure the parent surface observes the change.

Common valid approaches:

* shared store
* SwiftData query
* `ObservableObject` / `Observable`
* `onSave` callback
* repository-published change
* `NotificationCenter` only when it is the right tool for the context

Do not just save and dismiss while leaving the parent stale.

---

## 9. Async Operations Must Prevent Races

Async loading, saving, and refreshing must account for race conditions.

Requirements:

* UI updates must occur on `MainActor`
* old requests must not overwrite newer intent
* saves must not overwrite the latest database record with a stale in-memory object
* a long-open edit screen must confirm the entity still exists before saving
* rapid repeated interactions must not produce duplicate writes, duplicate notifications, or corrupted state

---

## 10. Error Handling Must Be Visible to the User

When a user-triggered write fails, do not rely on `print(error)` alone.

Provide visible user feedback such as:

* alert
* toast
* inline error
* form field validation error
* retry action

Save, delete, reminder, permission, import, migration, and database write failures must all expose an explicit error path.

Do not represent the operation as finally successful before persistence actually succeeds.

---

## 11. Time-Sensitive Screens Must Handle Date Changes

Pages involving the following concepts must handle day rollover, timezone changes, and foreground/background transitions:

* today's tasks
* weekly statistics
* next reminder
* calendar views
* streaks
* expiry state
* inventory reminders
* trend statistics

Consider at minimum:

* app entering foreground
* crossing midnight
* system timezone changes
* daylight saving time changes
* long background suspension

---

## 12. Inventory / Remaining Amount Must Stay Linked to Records

Inventory, remaining quantity, consumption, and replenishment data must stay consistent with the underlying business records.

Do not freely edit the current inventory number from multiple unrelated places.

Prefer a model such as:

```text
starting inventory + replenishment records - consumption records = current inventory
```

If a cached field is used, it still must guarantee:

* decrease after a consumption record is created
* restoration after undo
* recomputation after record deletion
* recomputation after dosage changes
* rollback after save failure

---

## 13. Create and Edit Paths Must Be Unambiguous

When saving a form, the code must clearly know whether the operation is a create or an update.

On edit, do not let default values overwrite existing fields unintentionally.

Before saving, confirm:

* whether an `entityId` exists
* whether the action is create or edit
* whether update means partial field mutation or full replacement
* whether untouched fields might be cleared by mistake
* whether default form values could pollute existing data

---

## 14. Persisted Model Changes Must Consider Migration

When changing persisted SwiftData, SQLite, or Codable structures, consider compatibility with older user data.

Check items such as:

* whether new fields have defaults
* whether renamed fields have a migration path
* whether enum additions or renames remain compatible with old data
* whether relationship changes could crash app launch
* whether old-version data can still be read

Do not casually delete or rename persisted fields.

---

## 15. Previews / Mocks Must Cover Failure States

UI validation must not rely only on ideal mock data.

At minimum, cover:

* empty data
* a single record
* multiple records
* long text
* deletion / invalid references
* loading
* save failure
* permission denied
* notifications unavailable
* database error
* cross-day state

---

## 16. Required Post-Write Self-Check

After every task involving data writes, the agent should output a brief self-check:

```md
### Write Impact Check

- Core entities affected:
- Screens affected:
- Derived data needing refresh:
- Notifications involved:
- Cache involved:
- Historical snapshots involved:
- Delete / invalid references involved:
- Cross-day / time change sensitivity:
- User-visible error handling:
- Test scenarios covered:
```

If any item cannot be confirmed, say so explicitly rather than silently assuming it is complete.

---

## 17. Overall Principle

Write-oriented functionality is not complete just because the current screen appears to work.

Validate the full chain:

```text
user action
→ validation
→ persistence write
→ state refresh
→ derived data recomputation
→ dependent surface sync
→ notification / cache sync
→ user-visible error feedback
→ failure-path testing
```

If this chain has not been checked end to end, the feature is not complete.
