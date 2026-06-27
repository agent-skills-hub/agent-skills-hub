---
name: sendmux-token-efficient-usage
description: "Choose low-token Sendmux routes with batching, snippets, sync deltas, pagination, ETags, and idempotency."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-token-efficient-usage
---

# Sendmux token-efficient usage

Use this skill to choose the lowest-cost Sendmux route that still answers the task correctly.

## When to Use This Skill

- Use when a Sendmux task could use MCP, CLI, SDK, or direct HTTP and the user needs the cheapest correct route.
- Use when choosing batch sends, mailbox search/count/batch reads, sync deltas, cursor pagination, ETags, or idempotency keys.
- Use when avoiding broad mailbox body reads, full mailbox scans, or broad log fetches.

## Copy-Paste Example

```text
User: "Find unread invoice mail with as few Sendmux calls as possible; count, search snippets, then batch-get selected IDs."
```

## Boundaries

- Do not ask the user to paste an API key.
- Use send-capable `smx_mbx_*` keys or owner-approved Sending-resource `smx_agent_*` tokens for Sending calls, and `smx_mbx_*` keys for normal Mailbox calls.
- Use scoped `smx_agent_*` only for the calls its scopes and resource allow. Pre-claim agent tokens cannot send.
- Use `smx_root_*` for Management calls.
- Do not default to MCP for every task. MCP is best when the required tool is curated; CLI and SDK cover broader surfaces.
- Do not read full mailbox bodies, every message, or every log row unless the user asks for full content and narrower calls cannot answer.

## Surface choice

| Situation                               | Use                                 | Why                                                           |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| Connected agent and curated tool exists | MCP tool                            | Small schema and no SDK boilerplate.                          |
| One-off terminal task                   | `sendmux` CLI with `--json`         | Direct, scriptable, exposes the full generated operation set. |
| Application code or repeated workflow   | SDK for the project already in use  | Reuses client setup, pagination, headers, and retry helpers.  |
| MCP lacks the needed operation          | CLI for terminal work, SDK for code | Do not invent uncurated MCP tools.                            |
| No package/tooling available            | Direct HTTP                         | Keep request bodies and headers aligned to OpenAPI.           |

## Cheapest-call map

| Task                            | Cheapest correct default                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Send one outbound email         | `sending_send_email`, CLI `sending:send`, SDK `sendingSendEmail`; include `Idempotency-Key`.                                           |
| Send multiple outbound emails   | `sending_send_email_batch`, CLI `sending:send:batch`, SDK `sendingSendEmailBatch`; do not loop single sends.                           |
| Count matching mailbox messages | `mailbox_count_messages`, CLI `mailbox:count-messages`, SDK `mailboxCountMessages`.                                                    |
| Search mailbox text             | `mailbox_search_message_snippets`, CLI `mailbox:search-message-snippets`, SDK `mailboxSearchMessageSnippets`; then fetch selected IDs. |
| Read several known messages     | `mailbox_batch_get_messages`, CLI `mailbox:batch-get-messages`, SDK `mailboxBatchGetMessages`.                                         |
| Update/delete several messages  | Batch update/delete after explicit confirmation.                                                                                       |
| Resume broad mailbox sync       | `mailbox_get_changes`, CLI `mailbox:get-changes`, SDK `mailboxGetChanges`.                                                             |
| Resume filtered mailbox sync    | CLI/SDK `mailbox:query-message-changes` / `mailboxQueryMessageChanges`; MCP does not curate it yet.                                    |
| Watch live mailbox events       | CLI/SDK `mailbox:stream-events` / `mailboxStreamEvents`; MCP does not curate it yet.                                                   |
| Scan threads                    | List threads, then fetch one thread or its messages.                                                                                   |
| Manage domains/mailboxes/keys   | Management MCP for curated create/list/get/update/suspend/resume/key tools; CLI/SDK for uncovered lifecycle work.                      |
| Manage sending accounts         | CLI/SDK; MCP does not curate provider tools yet.                                                                                       |
| Manage webhooks                 | MCP for list/create/test; CLI/SDK for get/update/delete/rotate/delivery payloads.                                                      |
| Inspect spend, logs, metrics    | Summary/metrics first; filter log lists with small `limit`, then fetch one selected row.                                               |

## Read less

For mailbox questions, reduce the result set before reading content:

1. Count when the user asks "how many" or when the query may be broad.
2. Search snippets with a small `limit` when the user needs examples.
3. Batch-get only selected message IDs.
4. Request clean body/content only when message text affects the answer.

CLI:

```bash
sendmux mailbox:count-messages \
  --query q=invoice \
  --query is_unread=true \
  --json

sendmux mailbox:search-message-snippets \
  --query q=invoice \
  --query is_unread=true \
  --query limit=10 \
  --json

sendmux mailbox:batch-get-messages \
  --body '{
    "ids": ["eml_abc", "eml_def"],
    "body_mode": "clean_json",
    "max_body_chars": 4000,
    "strip_quotes": true,
    "strip_signature": true,
    "include_attachments": "metadata"
  }' \
  --json
```

## Write fewer requests

Batch when there is more than one target.

```bash
sendmux sending:send:batch \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body-file ./messages.json \
  --json

sendmux mailbox:batch-update-messages \
  --body '{
    "ids": ["eml_abc", "eml_def"],
    "seen": true,
    "if_in_state": "state_from_prior_read"
  }' \
  --json
```

For batch sends, inspect every per-message result before reporting success. Batch can contain mixed outcomes.

## Sync by delta

Use sync endpoints instead of re-listing stable data.

Broad mailbox sync:

```bash
sendmux mailbox:get-changes \
  --query messages_since_state="$MESSAGES_STATE" \
  --query folders_since_state="$FOLDERS_STATE" \
  --query threads_since_state="$THREADS_STATE" \
  --query limit=100 \
  --json
```

Filtered message sync:

```bash
sendmux mailbox:query-message-changes \
  --query since_query_state="$QUERY_STATE" \
  --query q=invoice \
  --query is_unread=true \
  --query limit=100 \
  --json
```

Store the returned state token. Continue with the same filters only while `has_more` is true and the next page is needed.

## Transfer less

- Use small `limit` values on list calls.
- Follow `pagination.next_cursor` only until enough evidence has been gathered.
- Prefer summary or metrics endpoints before log lists.
- Use `If-None-Match` for repeated detail reads that previously returned an `ETag`.
- Use `If-Match` for updates when the prior read returned an `ETag`.

CLI conditional examples:

```bash
sendmux management:get-email-log \
  --path public_id=dlog_abc \
  --if-none-match "$ETAG" \
  --json

sendmux management:update-mailbox \
  --path public_id=mbx_abc \
  --if-match "$ETAG" \
  --body '{"display_name":"Agent Inbox"}' \
  --json
```

SDK helpers:

```
import {
  conditionalHeaders,
  idempotencyHeaders,
  paginate,
  responseEtag,
} from "@sendmux/core";

const headers = conditionalHeaders({ ifNoneMatch: priorEtag });
const writeHeaders = {
  ...conditionalHeaders({ etag: priorEtag }),
  ...idempotencyHeaders(operationKey),
};
```

## Retry safely

Use `Idempotency-Key` on supported mutations so retrying does not create duplicate work.

Good candidates:

- `sending:send` and `sending:send:batch`.
- `mailbox:send-message`.
- Management creates, mailbox key creation, suspend/resume, provider mutations, webhook create/rotate/test.

When retrying application code, prefer SDK retry helpers only for safe reads or idempotent writes. Non-idempotent writes should fail rather than risk duplicate side effects.

## Routing

- Setup, key scopes, first call: `sendmux-getting-started`.
- Email send bodies, attachments, SMTP-vs-HTTP choice: `sendmux-send-email`.
- Mailbox read/search/sync/triage/reply details: `sendmux-mailbox-agent`.
- Management domains, mailboxes, webhooks, billing, logs: `sendmux-management`.
- CLI syntax and profiles: `sendmux-cli`.
- MCP installation and client config: `sendmux-mcp-setup`.

## Limitations

- Does not invent uncurated MCP tools when CLI or SDK is the broader supported surface.
- Does not read full mailbox bodies, every message, or every log row unless narrower calls cannot answer the task.
- Some recommended routes can send or mutate state, so user confirmation and idempotency still matter.
