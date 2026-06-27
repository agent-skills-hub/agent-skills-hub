---
name: sendmux-management
description: "Manage Sendmux domains, mailboxes, keys, providers, webhooks, billing, logs, and metrics."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-management
---

# Sendmux management

Use this skill for team administration with an `smx_root_` key.

## When to Use This Skill

- Use when managing Sendmux account-level resources with an smx_root_ key.
- Use for domains, mailbox provisioning, mailbox API keys, providers, webhooks, billing, spend, delivery logs, incoming logs, or metrics.
- Use when choosing Management MCP, CLI, or SDK for account administration.

## Copy-Paste Example

```text
User: "Create a Sendmux mailbox for agent@example.com and a mailbox key, but confirm before any destructive changes."
```

## Boundaries

- Do not ask the user to paste an API key or one-time secret.
- Use `smx_root_` keys for Management API calls.
- Do not use management calls to read, triage, sync, or reply from a mailbox; route those tasks to `sendmux-mailbox-agent`.
- Treat create-key and webhook-secret responses as sensitive one-time values. Put them only in the user's chosen secret store or secure output path.
- Confirm destructive operations before deleting domains, mailboxes, mailbox keys, sending accounts, or webhooks.
- Confirm the target resource before suspend, resume, rotate-secret, or test-delivery operations.

## Surface choice

| Task | Preferred surface |
| --- | --- |
| Domains | MCP for list/create/get/zone-file/verify; CLI or SDK for update/delete/filter rules. |
| Mailboxes | MCP for list/create/get/update/suspend/resume/key create/key delete; CLI or SDK for delete and filters. |
| Sending accounts | CLI or SDK. MCP does not curate provider tools yet. |
| Webhooks | MCP for list/create/test; CLI or SDK for get/update/delete/rotate-secret/deliveries/payloads. |
| Billing and spend | MCP `management_get_spend_summary`; CLI or SDK for balance and transactions. |
| Outbound delivery logs and metrics | MCP for list/get logs and metrics; CLI or SDK also work. |
| Incoming logs | CLI or SDK. MCP does not curate incoming-log tools yet. |

For terminal work, use the `sendmux` CLI with `--json`. For application code, use `@sendmux/management` and `createManagementClient`.

## Efficient defaults

- Use small `limit` values on list endpoints and follow `pagination.next_cursor` only when more rows are needed.
- Use `Idempotency-Key` on create, suspend, resume, rotate-secret, and test mutations that accept it.
- Use `If-None-Match` for repeated detail reads that return `ETag`.
- Use `If-Match` for update or replace operations when the API exposes optimistic concurrency.
- For questions about counts or trends, use summary or metrics endpoints before log lists.
- For one email or webhook delivery, list with filters first, then fetch the selected record.

## Domains

Use domains for sending setup and hosted mailbox domains.

MCP tools:

| Action | Tool |
| --- | --- |
| List | `management_list_domains` |
| Create | `management_create_domain` |
| Inspect | `management_get_domain` |
| DNS records | `management_get_domain_zone_file` |
| Verify DNS | `management_verify_domain` |

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:create-domain \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body '{"domain":"example.com","mode":"send_receive"}' \
  --json

SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:get-domain-zone-file \
  --path public_id=mdom_abc \
  --json

SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:verify-domain \
  --path public_id=mdom_abc \
  --json
```

SDK:

```ts
import {
  createManagementClient,
  managementCreateDomain,
  managementGetDomainZoneFile,
  managementVerifyDomain,
} from "@sendmux/management";

const client = createManagementClient({ apiKey: process.env.SENDMUX_API_KEY! });

await managementCreateDomain({
  client,
  headers: { "Idempotency-Key": process.env.IDEMPOTENCY_KEY! },
  body: { domain: "example.com", mode: "send_receive" },
});

const zone = await managementGetDomainZoneFile({
  client,
  path: { public_id: "mdom_abc" },
});

await managementVerifyDomain({
  client,
  path: { public_id: "mdom_abc" },
});
```

Use CLI `management:update-domain` or SDK `managementUpdateDomain` with `If-Match` for mode upgrades. Confirm before delete.

## Mailboxes and keys

Create mailboxes with a root key; use mailbox keys afterwards for agent mailbox work.

MCP tools:

| Action | Tool |
| --- | --- |
| List | `management_list_mailboxes` |
| Create | `management_create_mailbox` |
| Inspect | `management_get_mailbox` |
| Update | `management_update_mailbox` |
| Suspend | `management_suspend_mailbox` |
| Resume | `management_resume_mailbox` |
| Create mailbox key | `management_create_mailbox_key` |
| Delete mailbox key | `management_delete_mailbox_key` |

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:create-mailbox \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body '{
    "email": "agent@example.com",
    "display_name": "Agent Inbox",
    "quota_bytes": 1073741824
  }' \
  --json

SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:create-mailbox-key \
  --path public_id=mbx_abc \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body '{"name":"agent-runtime"}' \
  --json
```

After creating a mailbox key, hand mailbox read/search/reply work to `sendmux-mailbox-agent` with the new `smx_mbx_` key.

Suspend keeps the mailbox and messages but blocks mailbox access. Delete removes the mailbox and revokes associated keys; confirm deletion explicitly.

## Sending accounts

Sending-account management is full CLI/SDK territory until MCP curates provider tools.

CLI commands:

| Action | Command |
| --- | --- |
| List | `sendmux management:providers:list --json` |
| Create | `sendmux management:create-provider --json` |
| Inspect | `sendmux management:get-provider --json` |
| Update | `sendmux management:update-provider --json` |
| Activate | `sendmux management:activate-provider --json` |
| Deactivate | `sendmux management:deactivate-provider --json` |
| Test | `sendmux management:test-provider --json` |
| Limits | `sendmux management:get-provider-limits --json` |
| Stats | `sendmux management:get-provider-stats --json` |
| Usage | `sendmux management:get-provider-usage --json` |

SDK helpers include `managementListProviders`, `managementCreateProvider`, `managementGetProvider`, `managementUpdateProvider`, `managementActivateProvider`, `managementDeactivateProvider`, `managementTestProvider`, `managementGetProviderLimits`, `managementGetProviderStats`, and `managementGetProviderUsage`.

Do not print provider credentials. Put secrets in the user's chosen secret store.

## Webhooks

MCP covers the common create-and-test path:

| Action | Tool |
| --- | --- |
| List | `management_list_webhooks` |
| Create | `management_create_webhook` |
| Test | `management_test_webhook` |

Use CLI or SDK for full lifecycle work:

| Action | CLI command | SDK helper |
| --- | --- | --- |
| Inspect | `management:get-webhook` | `managementGetWebhook` |
| Update | `management:update-webhook` | `managementUpdateWebhook` |
| Delete | `management:delete-webhook` | `managementDeleteWebhook` |
| Rotate secret | `management:rotate-webhook-secret` | `managementRotateWebhookSecret` |
| List deliveries | `management:list-delivery` | `managementListDelivery` |
| Get delivery payload | `management:get-delivery-payload` | `managementGetDeliveryPayload` |

Use `Idempotency-Key` on create, rotate-secret, and test. Store the webhook signing secret at creation or rotation time; it cannot be retrieved later.

## Billing, logs, and metrics

Use read-only summary endpoints before broad log reads.

| Question | Preferred call |
| --- | --- |
| Spend trend | `management_get_spend_summary`, CLI `management:get-spend-summary`, SDK `managementGetSpendSummary`. |
| Current balance | CLI `management:list-balance`, SDK `managementListBalance`. |
| Transaction history | CLI `management:list-transactions`, SDK `managementListTransactions`. |
| Outbound delivery logs | `management_list_email_logs`, CLI `management:list-email-logs`, SDK `managementListEmailLogs`. |
| One outbound log | `management_get_email_log`, CLI `management:get-email-log`, SDK `managementGetEmailLog`. |
| Delivery metrics | `management_get_email_metrics`, CLI `management:get-email-metrics`, SDK `managementGetEmailMetrics`. |
| Incoming logs | CLI `management:list-inbox-logs` / `management:get-inbox-log`, SDK `managementListInboxLogs` / `managementGetInboxLog`. |

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:get-spend-summary \
  --query days=30 \
  --json

SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:list-email-logs \
  --query status=failed \
  --query limit=20 \
  --json

SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux management:get-email-log \
  --path public_id=dlog_abc \
  --header If-None-Match="$ETAG" \
  --json
```

## Routing

- First setup/auth check: `sendmux-getting-started`.
- Single or batch transactional sending: `sendmux-send-email`.
- Mailbox read, search, sync, triage, or reply: `sendmux-mailbox-agent`.
- CLI command details: `sendmux-cli`.
- MCP connection setup: `sendmux-mcp-setup`.
- Cheapest-call doctrine: `sendmux-token-efficient-usage`.

## Limitations

- Does not read, triage, sync, or reply from a mailbox; the source routes that to sendmux-mailbox-agent.
- Requires smx_root_ credentials for Management API calls.
- Requires confirmation before destructive operations, suspend, resume, rotate-secret, or test-delivery operations.
