# Silent Fallback and Error-Handling Review

## 1. Principle

A fallback is not automatically wrong. It becomes dangerous when it invents, rewrites, or hides a business fact after the authoritative source failed or was missing.

The governing rule is:

> Preserve the distinction between valid empty state, missing required data, query failure, migration state, and display-only degradation.

## 2. Prohibited fallback

### Business-fact fallback

Examples:

- reconstructing a missing execution record from the latest configuration
- assigning a default relationship when a stable identifier is absent
- rebuilding historical data from a mutable current entity
- creating substitute persisted objects after a repository failure
- using a second data source whenever the primary source returns no rows without distinguishing error from empty

### Error-swallowing fallback

Examples:

```swift
let value = try? repository.load()
return value ?? []
```

```swift
do {
    try save()
} catch {
    return defaultModel
}
```

These patterns may be acceptable for optional decoration, but not when they hide a failed business operation.

### Default-value masking

A default is unsafe when the field is required to preserve identity, ordering, schedule position, ownership, or historical meaning.

Examples:

- missing anchor becomes the first cycle/day
- missing permission becomes “disabled” without explanation
- failed query becomes “no records”
- missing relationship becomes the first or nearest matching entity

### Long-lived dual source

Do not keep new and legacy facts in normal reads indefinitely:

```swift
if newRecords.isEmpty {
    return buildFromLegacyConfiguration()
}
```

An empty new result may be valid. Migration must not live forever inside ordinary reads.

## 3. Allowed categories

### Controlled migration fallback

Temporarily allowed only when all are true:

- explicit migration phase and identifier
- deterministic eligibility
- idempotent transformation
- audit result
- failure reporting
- removal plan
- no overwrite of newer user data

### Display degradation

Allowed when it changes presentation but not facts, for example:

- placeholder icon
- generic nonbusiness subtitle
- hiding optional decoration
- accessible text substitute for a missing image

The UI must not imply a false successful business state.

### Safe stop

When required data is unavailable, a safe stop is often correct:

- disable the action
- show a recoverable error
- allow retry
- require user confirmation
- mark the record for migration or repair

## 4. Repository contracts

A repository must distinguish:

- successful result containing zero entities
- entity not found
- permission or access denial
- persistence/query failure
- data invariant violation

Prefer throwing, typed result, or an explicit domain error. Do not return `[]` or `nil` for every failure category.

## 5. Domain errors

Use errors that describe the violated contract, for example:

```swift
enum DataIntegrityError: Error {
    case missingRequiredIdentifier
    case missingRequiredRelationship(entityID: String)
    case invalidHistoricalSnapshot(entityID: String)
    case migrationRequired(entityID: String)
    case inconsistentDerivedState(entityID: String)
}
```

Names and payloads must be adapted to the project and privacy policy.

## 6. View-layer boundary

A view may:

- render loading, empty, error, and retry states
- choose visual placeholders
- collect user confirmation

A view must not:

- create missing domain records
- repair persistence relationships
- infer identity from labels, dates, order, or proximity
- bypass the repository/application service
- convert a failed write into a visually successful state

## 7. Migration

Migration should run through:

- schema migration
- application migration coordinator
- controlled one-time upgrade task
- explicit diagnostic/repair tool

Migration must be:

- idempotent
- resumable or safely restartable
- auditable
- bounded in scope
- removable after completion

At minimum record migration ID, entity type/ID, result, failure reason, and timestamp. Avoid unnecessary sensitive content.

## 8. Observability

Replace removed fallback with appropriate evidence:

- typed error
- structured logging
- debug assertion where safe
- retry/recovery state
- regression test

Useful structured fields include subsystem, operation, entity identifier, expected state, actual state, and error type.

## 9. Review search patterns

Search candidates:

- `try?`
- `?? []`
- `?? defaultValue`
- empty `catch`
- catch followed by continued success flow
- `if newData.isEmpty { useLegacyData() }`
- `newValue ?? oldValue`
- reconstruction of historical data from current configuration
- matching entities by name, time, array index, or nearest value
- creating substitute data after a failed query
- returning an empty collection from a failed repository operation
- production `assertionFailure` followed by continued invalid flow
- permanent new/legacy dual reads

A match is only a candidate. Classify it as:

- must remove
- migration-only with removal requirement
- valid display degradation
- unrelated to business facts
- requires product decision

## 10. Acceptance

A fallback correction is complete when:

- authoritative source is explicit
- valid empty and error are distinguishable
- no business fact is guessed
- migration, when required, is separate and idempotent
- user recovery is defined
- logging respects privacy
- tests cover missing data, failure, and valid empty state
