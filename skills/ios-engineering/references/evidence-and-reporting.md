# Evidence and Reporting Standard

## Evidence hierarchy

Prefer evidence in this order:

1. deterministic test that exercises the owning business layer
2. integration or persistence test across the changed boundary
3. target build result
4. device or simulator reproduction
5. visual comparison for UI-only claims
6. static code inspection when execution is unavailable

Static inspection alone cannot prove runtime, migration, notification delivery, or visual behavior.

## Finding evidence

A review finding must contain:

- path or component
- triggering state
- reachable behavior
- expected behavior
- impact
- correction
- regression test recommendation

## Completion evidence

For every completion criterion, record:

- evidence type
- test, command, path, or scenario
- result
- limitations

## Prohibited claims

Avoid:

- “should be fixed” without validation
- “tests pass” without naming the executed entry
- “fully complete” when device, permission, migration, or visual checks are still missing
- “no issue” based only on a narrow happy-path test
