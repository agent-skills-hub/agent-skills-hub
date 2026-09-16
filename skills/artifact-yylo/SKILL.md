---
name: artifact-yylo
description: Capture and retrieve durable YYLO Ledger artifact Records with intentional profiles, payload modes, provenance, retention, and secret-safe immutable evidence.
risk: safe
source: https://github.com/yylo-dev/yylo-skills/tree/main/skills/artifact-yylo
---

# Use YYLO artifact Records

## Overview

Durable evidence — reports, receipts, model output, bounded process output — is captured as immutable YYLO Ledger artifact Records with explicit profile, payload mode, provenance, and retention.

## When to Use This Skill

- Use when a user asks to capture, store, or retrieve run evidence or generated reports in a YYLO project.
- Use when later verification depends on exact bytes (choose an immutable payload mode).
- Do not treat an artifact Record as a release artifact or as authority to publish or deploy.

Treat Ledger as the source of truth for artifact identity and metadata. Use `yy ledger` in a YYLO controller or `yylo-ledger` standalone. Inspect `COMMAND artifact --help`; if unavailable, do not create store files manually.

## Classify before capture

Choose the profile matching the evidence:

- `stdout`: bounded process output;
- `model-output`: an agent/model response;
- `report`: a generated human- or machine-readable result;
- `receipt`: evidence binding an operation and its inputs/outcome.

Choose payload mode deliberately:

- `inline`: small immutable bytes embedded in the Record;
- `local`: immutable content-addressed bytes in Ledger storage;
- `external`: immutable external bytes with URI, digest, and size;
- `link`: URI reference without an immutable-byte guarantee.

Prefer immutable evidence when later verification depends on exact bytes. A link must never be presented as content-addressed proof.

## Create explicitly

Use file/stdin transport and provide the media type:

```bash
yy ledger artifact create --title "Focused test report" --profile report \
  --mode local --media-type application/json --file report.json
```

For external immutable content, provide the supported URI, SHA-256 digest, and size shown by installed help. Never embed URI credentials. Ledger rejects unsafe schemes, traversal, size/digest mismatches, oversized capture, and known secret patterns.

Attach only supported, non-secret provenance such as actor, agent, model, session, run, invocation, task, or workflow identity. Task/workflow provenance uses immutable Record IDs. Select `temporary`, `standard`, or `permanent` retention deliberately; retention metadata does not itself authorize deletion.

## Find and verify

```bash
yy ledger artifact search --profile report --projection summary --limit 20 -f json
yy ledger artifact get RECORD_ID -f json
yy ledger artifact history RECORD_ID -f ndjson
```

Use bounded metadata/summary projections before requesting payload details. Verify profile, mode, media type, digest, size, provenance, retention, revision, and immutable ID before relying on evidence.

Artifact payloads are immutable evidence. Represent replacement with explicit predecessor/successor relationships and the installed revision-safe update contract; do not overwrite bytes or edit content objects. Archive is a lifecycle transition, not deletion. Release, publication, external upload, retention execution, and production mutation always require separate authority.
