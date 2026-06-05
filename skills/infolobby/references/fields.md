# InfoLobby field types — record-write quirks

How values are written and read for each field type. Field metadata comes from `GET /table/<tid>/get`.

Field `id` is the slug used in record payloads. The display label is in `name` — do **not** use it as a key.

## Lookup

A pointer to a record in another table.

- **Write:** send the related record id as an integer. `"customer": 1`.
- Also accepts `{id,title}` for editing-UI parity: `"customer": {"id": 1, "title": "Acme"}`. Only the `id` is stored.
- **Read (`record/get`):** inflated to `{"id": 1, "title": "Acme"}`.
- **Read (`records/query`):** scalar string id `"customer": "1"` with a sibling `"customer.json"` holding the title — query returns flat key/value rows.

Target table is declared on the schema entry as `"table":{"id":<tid>,"text":"<Table Name>"}`. Skill code does not need to set this; only `create-table` flows do.

## User

A workspace member.

- **Write:** the member's **email address**. `"agent": "admin@example.com"`.
- Numeric `user_id` is rejected with `Invalid User N for Agent`, even when the value came back from `/space/<sid>/members/list`.
- Object shapes (`{id,name}`, `{user_id}`) return an empty 200 body and silently fail to create a record — always check the response body is non-empty.

## Number

- Integer by default. `49.99` is truncated to `50`.
- The field's `decimals` (in `options.decimals` on read; `decimals` at top level on schema write) controls precision.
- With `decimals` set, accepts JSON numbers or numeric strings; preserved verbatim.

## Select

- **Write:** a single string. `"status": "Active"`.
- **Read (`record/get`):** comes back as a one-element array `"status": ["Active"]`.
- **Read (`records/query`):** scalar string.
- Reading the schema, `options` is a normalized nested object — do not feed it back verbatim if reusing for create-table.

## Date

- **Write:** ISO date string `"YYYY-MM-DD"`.
- **Filter values** can also use the date tokens listed in `endpoints.md` (`today`, `start_of_month`, `-7d`, etc.).

## Key (auto-id)

- One per table. Read-only on the API for records; the record id is auto-assigned.
- `record/get` returns the int at the top level as well as `data.<key_field>` (often as string).

## Calc

- Read-only computed column. Writes return `Unassignable - cannot set calculation field: <id>` — omit from `data`.
- Aggregate calcs over empty related sets return `null`. Compose with `IFNULL(...,0)` in formulas (formula authoring is out of skill scope).

## File

- A list of attachments. Manage through the `/files/*` endpoints — not by editing `data` directly.
- Each attachment is `{name, path, type, size, host}`. `path` is server-assigned.
- Replacing/reordering: use record update with the full attachment array.

## Common write failures

| Symptom | Cause |
|---|---|
| `Unauthorized - no field edit permissions for field: X` | Wrong field id slug. Refetch schema. |
| `Unknown column '' in 'field list'` | Empty `data` object. |
| `Unassignable - cannot set calculation field: X` | Wrote to a calc field. Drop it from `data`. |
| `Invalid User N for Agent` | User field got an id instead of an email. |
| empty 200 body on create | User field got an object shape; record was not created. |
| values silently truncate decimals | Number field has no `decimals` set. |
