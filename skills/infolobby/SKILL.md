---
name: infolobby
description: Use when the user wants to read, query, or modify their InfoLobby workspace data — list spaces and tables, get a table schema, create/update/delete/query records, post comments, or upload/download record attachments. Requires the user's InfoLobby API key, which the skill prompts for once and caches at ~/.infolobby/key.
license: MIT
risk: safe
source: https://infolobby.com/skill
metadata:
  vendor: Globi Web Solutions
---

# InfoLobby

Access an InfoLobby account from any agent using plain `curl`. No SDK.

## When to Use

Use for InfoLobby data operations: discover spaces/tables, inspect schemas, query records, mutate records, comments, and record attachments.

## Setup

The API key lives at `~/.infolobby/key` (one line, the raw token).

**On every invocation**, first try to read it:

```bash
INFOLOBBY_KEY="$(cat ~/.infolobby/key 2>/dev/null)"
INFOLOBBY_API="https://infolobby.com/api"
```

If `~/.infolobby/key` does not exist, ask the user once for their InfoLobby API key (`il_live_…` or `il_user_…`), then write it. Do not log or echo the key back.

- macOS / Linux:
  ```bash
  mkdir -p ~/.infolobby
  ( umask 077 && printf '%s' "<KEY>" > ~/.infolobby/key )
  ```
- Windows (PowerShell):
  ```powershell
  New-Item -ItemType Directory -Force "$HOME\.infolobby" | Out-Null
  Set-Content -NoNewline "$HOME\.infolobby\key" '<KEY>'
  ```

Where the user gets a key:

- **Account key** (`il_live_…`): InfoLobby → **Account → API Keys**. Account-owner only. Acts as the owner across all in-scope workspaces.
- **Personal key** (`il_user_…`): InfoLobby → **Profile → Personal API Keys**. Acts as that user; cannot manage workspaces or table schemas.

Send the key on every request as `Authorization: Bearer $INFOLOBBY_KEY`.

## Scope

This skill covers data work only:

- discover: list spaces, list tables, fetch a table schema
- records: create, get, update, delete, batch-delete, **query with filters / saved views**
- comments: list, post, attach files
- files: upload, download, delete record attachments (base64 in JSON, 50 MB cap)

This skill does **not** create/modify workspaces, create/modify table schemas, manage members, run flows, or touch billing. Point the user at [https://infolobby.com/api-docs](https://infolobby.com/api-docs) for those.

## Workflow

Always discover before mutating:

1. `GET /spaces/list` → pick a space.
2. `GET /space/<space_id>/tables/list` → pick a table.
3. `GET /table/<table_id>/get` → read field `id` slugs from the schema. Record payloads use these slugs, **not** display names — a wrong key returns `Unauthorized - no field edit permissions for field: X`.

## Cheat sheet

| Action | Verb | Path |
|---|---|---|
| List spaces | GET | `/spaces/list` |
| List tables | GET | `/space/<sid>/tables/list` |
| Get table schema | GET | `/table/<tid>/get` |
| List views | GET | `/table/<tid>/views/list` |
| Query records | POST | `/table/<tid>/records/query` |
| Get record | GET | `/table/<tid>/record/<rid>/get` |
| Create record | POST | `/table/<tid>/records/create` |
| Update record | POST | `/table/<tid>/record/<rid>/update` |
| Delete record | POST | `/table/<tid>/record/<rid>/delete` |
| List comments | GET | `/table/<tid>/record/<rid>/comments/get?limit=20` |
| Add comment | POST | `/table/<tid>/record/<rid>/comments/create` |
| Upload attachment | POST | `/table/<tid>/record/<rid>/files/create` |
| Download attachment | POST | `/table/<tid>/record/<rid>/files/get` |
| Delete attachment | POST | `/table/<tid>/record/<rid>/files/delete` |

## Query example

```bash
curl -s "$INFOLOBBY_API/table/101/records/query" \
  -H "Authorization: Bearer $INFOLOBBY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "fields": ["company_name","email","status"],
    "filters": [{"column":"status","compare":"=","value":"Active"}],
    "order_by": "company_name",
    "order_dir": "A",
    "limit": 50
  }'
```

Pass `"view_id": N` to query through a saved view; the view's filters and sort replace the request's.

## Create / update record example

```bash
curl -s -X POST "$INFOLOBBY_API/table/101/records/create" \
  -H "Authorization: Bearer $INFOLOBBY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"data":{"company_name":"Acme","email":"hello@acme.test","status":"Active"}}'
```

`data` keys must be field `id` slugs from `GET /table/<tid>/get`. Update bodies are partial — include only fields to change.

## Files (base64)

```bash
B64=$(base64 -w0 invoice.pdf)
curl -s -X POST "$INFOLOBBY_API/table/101/record/5/files/create" \
  -H "Authorization: Bearer $INFOLOBBY_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"field_id\":\"attachments\",\"name\":\"invoice.pdf\",\"type\":\"application/pdf\",\"data\":\"$B64\"}"
```

Decoded payload must be ≤ 50 MB. Download returns base64 in `data`. Attachment object shape on the record is `{name,path,type,size,host}`. `files/delete` strips by `(field_id, path)`.

## Error handling

- Non-2xx → plain text body.
- `401` → key missing/invalid/revoked; reprompt and rewrite `~/.infolobby/key`.
- `403` → read-only key on a write call, or endpoint not in scope for the key type.
- `429` → respect `Retry-After`. `X-RateLimit-*` headers describe the bucket.
- `500` with `Unauthorized - no field edit permissions for field: X` → field slug is wrong; refetch schema.

## References

- [references/endpoints.md](references/endpoints.md) — full endpoint list and request bodies for the in-scope surface.
- [references/fields.md](references/fields.md) — field-type quirks that surprise first-time callers (lookup, select, user, number, date).

## Tips

- Base URL: `https://infolobby.com/api`.
- Responses are JSON for data endpoints; errors are plain text with non-2xx status.
- Use `curl -i` when debugging auth or scoping issues.
- Personal keys (`il_user_…`) cannot CRUD workspaces or table schemas — that is by design.
- Never write the user's key into chat output, command history echoes, or logs.
