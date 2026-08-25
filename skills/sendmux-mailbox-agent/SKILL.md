---
name: sendmux-mailbox-agent
description: "Read, search, sync, triage, mutate, or reply from one Sendmux mailbox efficiently."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-mailbox-agent
---

# Sendmux mailbox agent

Use this skill for mailbox-scoped workflows with an `smx_mbx_` key or scoped `smx_agent_` token: read, search, triage, reply when allowed, and sync one mailbox.

## When to Use This Skill

- Use when reading, searching, counting, syncing, triaging, filing, deleting, threading, or replying from one Sendmux mailbox.
- Use when the user has an smx_mbx_ key or scoped smx_agent_ token for mailbox runtime work.
- Use when minimizing mailbox reads with count, snippet, batch-get, thread, folder, or sync calls.

## Copy-Paste Example

```text
User: "Search the Sendmux mailbox for unread invoice messages, show snippets first, and ask before marking anything read."
```

## Boundaries

- Do not ask the user to paste an API key.
- Do not use a root key for mailbox work.
- Do not create mailboxes or mailbox keys here; route those tasks to `sendmux-management`.
- Do not delete or mutate messages without explicit user confirmation.
- Treat pre-claim `smx_agent_` tokens as read/receive only. They do not have `email.send`; owner-approved app-resource `smx_agent_` tokens may send from their assigned mailbox when the token includes `email.send`.
- If a credential grants more than one mailbox, include `mailbox_id` on mailbox calls; otherwise omit it.

## Efficient defaults

| Task                               | Preferred call                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Identify the mailbox               | `mailbox_get_me`, CLI `mailbox:me:get`, SDK `mailboxGetMe`.                                                   |
| Count matching messages            | `mailbox_count_messages`, CLI `mailbox:count-messages`, SDK `mailboxCountMessages`.                           |
| Search text                        | `mailbox_search_message_snippets`, CLI `mailbox:search-message-snippets`, SDK `mailboxSearchMessageSnippets`. |
| Read known IDs                     | `mailbox_batch_get_messages`, CLI `mailbox:batch-get-messages`, SDK `mailboxBatchGetMessages`.                |
| Mark, flag, or label many messages | `mailbox_batch_update_messages`, CLI `mailbox:batch-update-messages`, SDK `mailboxBatchUpdateMessages`.       |
| Delete many messages               | `mailbox_batch_delete_messages`, CLI `mailbox:batch-delete-messages`, SDK `mailboxBatchDeleteMessages`.       |
| Reply/send from this mailbox       | `mailbox_send_message`, CLI `mailbox:send-message`, SDK `mailboxSendMessage`.                                 |
| Threads                            | `mailbox_list_threads`, `mailbox_get_thread`, `mailbox_list_thread_messages`.                                 |
| Folders                            | `mailbox_list_folders`; inspect folders before filing or moving messages.                                     |
| Broad sync                         | `mailbox_get_changes`, CLI `mailbox:get-changes`, SDK `mailboxGetChanges`.                                    |
| Filtered message sync              | CLI/SDK `mailbox:query-message-changes` / `mailboxQueryMessageChanges`. MCP does not curate this yet.         |
| Live mailbox events                | CLI/SDK `mailbox:stream-events` / `mailboxStreamEvents`. MCP does not curate this yet.                        |

## Search before reading

Use this sequence for most "find messages about X" tasks:

1. Count first when the user asks "how many" or when a broad search may be large.
2. Use search snippets with a small `limit`.
3. Batch-get only the selected message IDs.
4. Read clean body/content only for messages whose content matters.

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:count-messages \
  --query q=invoice \
  --query is_unread=true \
  --json

SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:search-message-snippets \
  --query q=invoice \
  --query is_unread=true \
  --query limit=10 \
  --json

SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:batch-get-messages \
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

Use the same commands by putting a scoped `smx_agent_` token in `SENDMUX_API_KEY` when the operation is within its scopes.

SDK:

```ts
import {
  createMailboxClient,
  mailboxBatchGetMessages,
  mailboxCountMessages,
  mailboxSearchMessageSnippets,
} from "@sendmux/mailbox";

const client = createMailboxClient({ apiKey: process.env.SENDMUX_API_KEY! });

const count = await mailboxCountMessages({
  client,
  query: { q: "invoice", is_unread: true },
});

const snippets = await mailboxSearchMessageSnippets({
  client,
  query: { q: "invoice", is_unread: true, limit: 10 },
});

const messages = await mailboxBatchGetMessages({
  client,
  body: {
    ids: snippets.data.snippets.map((item) => item.message_id),
    body_mode: "clean_json",
    max_body_chars: 4000,
    strip_quotes: true,
    strip_signature: true,
    include_attachments: "metadata",
  },
});
```

## Triage and mutation

Use batch mutations for more than one message. Get user confirmation first.

Mark or label messages:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:batch-update-messages \
  --body '{
    "ids": ["eml_abc", "eml_def"],
    "seen": true,
    "flagged": false,
    "keywords": {
      "agent_triaged": true,
      "needs_reply": false
    },
    "if_in_state": "state_from_prior_read"
  }' \
  --json
```

Delete messages only after explicit confirmation:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:batch-delete-messages \
  --body '{
    "ids": ["eml_abc", "eml_def"],
    "permanent": false,
    "if_in_state": "state_from_prior_read"
  }' \
  --json
```

`permanent: false` moves messages to Trash. Treat `permanent: true` as irreversible and ask for explicit confirmation.

## Reply or send from the mailbox

Before composing, read the identity:

```text
mailbox_get_identity
```

Then send from the authenticated mailbox. Use `Idempotency-Key` for retries.

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux mailbox:send-message \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body '{
    "to": [{ "email": "user@example.com", "name": null }],
    "subject": "Re: Your message",
    "html_body": "<p>Thanks for the update.</p>",
    "text_body": "Thanks for the update."
  }' \
  --json
```

Mailbox send uses `to` as an array. `subject` and `to` are required. `from` is optional when sending from the authenticated mailbox identity.

Attachments may be inline small files:

```json
{
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "content": "<<base64-content>>"
}
```

or uploaded blobs:

```json
{
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "blob_id": "blob_..."
}
```

## Threads and folders

- Use `mailbox_list_threads` with small `limit` for conversation-level scanning.
- Use `mailbox_get_thread` for one thread summary.
- Use `mailbox_list_thread_messages` for message summaries in a known thread.
- Use clean content/body tools only for selected messages or threads.
- Use `mailbox_list_folders` before moving or filing messages.

CLI examples:

```bash
sendmux mailbox:list-threads --query q=renewal --query limit=10 --json
sendmux mailbox:list-thread-messages --path thread_id=thr_abc --query limit=20 --json
sendmux mailbox:folders:list --query limit=100 --json
```

## Sync

Use sync endpoints instead of re-listing the mailbox.

Broad mailbox sync:

```bash
sendmux mailbox:get-changes \
  --query types=messages,folders,threads \
  --query messages_since_state="$MESSAGES_STATE" \
  --query folders_since_state="$FOLDERS_STATE" \
  --query threads_since_state="$THREADS_STATE" \
  --query limit=100 \
  --json
```

Filtered message-list sync:

```bash
sendmux mailbox:query-message-changes \
  --query since_query_state="$QUERY_STATE" \
  --query q=invoice \
  --query is_unread=true \
  --query calculate_total=true \
  --query limit=100 \
  --json
```

Store the returned new state token. Follow `has_more` with the same filters when more changes remain.

## Error handling

- `401`: missing, invalid, or revoked key.
- `403`: wrong key surface or missing mailbox permission.
- `404`: selected message, thread, or folder does not exist in this mailbox.
- `409`: stale state or idempotency conflict; re-read state before mutating.
- `429` or `503`: retry according to response headers.

## Routing

- First setup/auth check: `sendmux-getting-started`.
- Independent outbound sending or batch sends: `sendmux-send-email`.
- Domain/mailbox/key creation and mailbox admin: `sendmux-management`.
- CLI command details: `sendmux-cli`.
- Cheapest-call doctrine: `sendmux-token-efficient-usage`.

## Limitations

- Does not create mailboxes or mailbox keys; the source routes those tasks to sendmux-management.
- Does not use root keys for mailbox work.
- Requires explicit confirmation before message deletion, mutation, or mailbox sending.
