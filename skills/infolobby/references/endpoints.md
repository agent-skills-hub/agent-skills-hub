# InfoLobby endpoints — public skill scope

Base URL: `https://infolobby.com/api`

Send `Authorization: Bearer <key>` on every request.

## Discovery

- `GET /spaces/list` — list spaces the key can see.
- `GET /space/<space_id>/get` — one space's metadata.
- `GET /space/<space_id>/tables/list` — tables in a space.
- `GET /table/<table_id>/get` — full table metadata + field schema. Use this to get field `id` slugs before any record write.
- `GET /table/<table_id>/views/list` — saved views (includes synthetic Default view id `0`).
- `GET /table/<table_id>/view/<view_id>/get` — one view's filters and sort.

## Records

Field write rules (see also `fields.md`):

- Keys in `data` are field `id` slugs from the schema, not display labels.
- Update is partial — send only fields to change.
- Empty `{"data":{}}` returns `Unknown column '' in 'field list'`.

| Verb | Path | Body |
|---|---|---|
| POST | `/table/<tid>/records/create` | `{"data":{...}}` |
| GET | `/table/<tid>/record/<rid>/get` | — |
| POST | `/table/<tid>/record/<rid>/update` | `{"data":{...}}` |
| POST | `/table/<tid>/record/<rid>/delete` | — |
| POST | `/table/<tid>/records/delete_batch` | `{"record_ids":[...]}` |
| POST | `/table/<tid>/records/query` | query body, see below |

### Query body

```json
{
  "fields": ["name","email"],
  "where": {"status": "Active"},
  "filters": [{"column": "age", "compare": ">", "value": 21}],
  "order_by": "name",
  "order_dir": "A",
  "search": "acme",
  "limit": 50,
  "offset": 0,
  "view_id": 0
}
```

Compare operators: `=`, `!=`, `<`, `<=`, `>`, `>=`, `contains`, `starts_with`, `ends_with`, `is_empty`, `is_not_empty`. Aliases: `EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `C`, `SW`, `EW`, `EMPTY`, `NEMPTY`.

Date tokens (use inside filter `value`): `today`, `now`, `yesterday`, `start_of_week`, `start_of_month`, `start_of_year`, `start_of_last_week`, `start_of_last_month`, `start_of_last_year`, and relative offsets `-Nd`/`+Nd`/`-Nw`/`+Nw`/`-Nm`/`+Nm`/`-Ny`/`+Ny`.

When `view_id` is set, the saved view's filters and sort **replace** the request's filters/sort. `search`, `limit`, `offset`, and `fields` still apply on top.

### Response shapes

`records/create`, `record/<rid>/get`, `record/<rid>/update` return `{"id","title","data":{…}}` with values nested under `data`. Select values come back as one-element arrays here: `"status":["Active"]`.

`records/query` returns a **flat array** of objects with field ids at the top level — no `data` wrapper. Select values are scalars. Lookup values are stringified ids with a sibling `<field>.json` of the title.

## Comments

- `GET /table/<tid>/record/<rid>/comments/get?limit=20&offset=0`
- `POST /table/<tid>/record/<rid>/comments/create` — body `{"content":"...", "attachments":[]}`
- `POST /table/<tid>/record/<rid>/comments/upload` — body `{"name","type","data"}`, base64; returns attachment metadata to put into `attachments` of the next comment create

Mentions inside `content` use `@{<user_id>:<display_name>}`. API-created comments show as `API: <key name>` in the UI.

## Files

JSON + base64 transport only (no multipart). 50 MB decoded cap.

- `POST /table/<tid>/record/<rid>/files/create` — body `{"field_id","name","type?","data"}`. Appends to the named file field. Allowed for non-read-only keys with edit rights on the field.
- `POST /table/<tid>/record/<rid>/files/delete` — body `{"field_id","path"}`. Strips the matching attachment from the field's array. Underlying storage is reclaimed by a nightly GC after a 30-day grace period (managed storage only).
- `POST /table/<tid>/record/<rid>/files/get` — body `{"field_id","path"}`. Returns `data` as base64. Allowed for read-only keys.

Attachment object stored on the record: `{name, path, type, size, host}`. The `path` is the tenant-prefixed S3 key; do not synthesise it — use the value returned by `create` or read from the record.

## Auth & limits — quick reference

- `il_live_` (account key): account rate bucket; can manage data across in-scope workspaces.
- `il_user_` (personal key): user permissions; ¼ of the account's default rate; cannot manage workspaces or table schemas.
- Read-only keys cannot write.
- IP allowlist failures return 401.

Rate-limit headers on every response: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `X-RateLimit-Scope`. On 429, honour `Retry-After`.

Error bodies are plain text. Status codes: 401 auth, 402 plan limit, 403 forbidden/scope, 429 rate limit, 500 application/validation, 503 API kill-switch.

## Out of scope

The following exist in the full public API but are intentionally **not** documented in this skill:

- workspaces CRUD, table schema CRUD, members
- subscriptions, standalone tasks, notifications, email
- flows, webforms, integrations

If the user needs any of those, point them at [https://infolobby.com/api-docs](https://infolobby.com/api-docs).
