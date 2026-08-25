---
name: sendmux-cli
description: "Use the Sendmux CLI for profiles, key-scope checks, JSON output, and generated operation commands."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-cli
---

# Sendmux CLI

Use this skill when the terminal is the right Sendmux surface.

## When to Use This Skill

- Use when the terminal is the right Sendmux surface.
- Use when the user needs exact sendmux CLI install, profile, flag, or generated operation command syntax.
- Use when preflighting key scope or producing machine-readable --json output.

## Copy-Paste Example

```text
sendmux mailbox:search-message-snippets --profile mailbox --query q=invoice --query limit=10 --json
```

## Boundaries

- Do not ask the user to paste API keys.
- Use `smx_root_` keys only for `management:*` commands.
- Use `smx_mbx_` keys or scoped `smx_agent_` tokens for `mailbox:*` commands.
- Use a send-capable `smx_mbx_` key or owner-approved Sending-resource `smx_agent_` token for `sending:*` commands. Pre-claim `smx_agent_` tokens cannot send.
- Do not run destructive commands without explicit confirmation.
- Use `--json` for agent-readable output.
- Prefer task-specific Sendmux skills when the user needs strategy; use this skill for exact CLI mechanics.

## Install

```bash
npm install -g @sendmux/cli
sendmux --help
```

The package exposes the `sendmux` binary.
Use the latest CLI before using `smx_agent_` tokens; older installs may reject that prefix before sending a request.

## Profiles

Create separate profiles for root and mailbox keys.

```bash
sendmux profiles:set default --api-key "$SENDMUX_ROOT_KEY" --default --json
sendmux profiles:set mailbox --api-key "$SENDMUX_MBX_KEY" --json
sendmux profiles:set sending --api-key "$SENDMUX_MBX_KEY" --json
sendmux profiles:list --json
sendmux profiles:show default --json
```

Profile reads mask stored keys. `profiles:set` reports `key_kind` as `root` or `mailbox`.

Authentication resolution:

1. `--api-key`, then `SENDMUX_API_KEY`.
2. If no direct key is present, `--profile` / `-p`, then `SENDMUX_PROFILE`, then the configured default profile.
3. Base URL comes from `--base-url`, then `SENDMUX_BASE_URL`, then the selected profile.

## Preflight

The CLI infers key kind from the prefix before sending a request.

| Command surface | Required key                                                                      |
| --------------- | --------------------------------------------------------------------------------- |
| `management:*`  | `smx_root_`                                                                       |
| `mailbox:*`     | `smx_mbx_` or scoped `smx_agent_`                                                 |
| `sending:*`     | Send-capable `smx_mbx_` key or owner-approved Sending-resource `smx_agent_` token |

Wrong-key examples fail before network:

```text
Command requires a root API key, but --api-key contains a mailbox API key.
Command requires a send-capable `smx_mbx_` key or owner-approved Sending-resource `smx_agent_` token, but --api-key contains a root API key.
```

## Command catalogue

The CLI exposes generated operation commands:

| Surface    | Count | Examples                                                                                                                                                   |
| ---------- | ----: | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Management |    52 | `management:domains:list`, `management:create-domain`, `management:create-mailbox`, `management:get-spend-summary`, `management:create-webhook`            |
| Mailbox    |    40 | `mailbox:search-message-snippets`, `mailbox:batch-get-messages`, `mailbox:query-message-changes`, `mailbox:send-message`, `mailbox:list-granted-mailboxes` |
| Sending    |     3 | `sending:get-open-api-spec`, `sending:send`, `sending:send:batch`                                                                                          |
| Profiles   |     3 | `profiles:list`, `profiles:set`, `profiles:show`                                                                                                           |

Use command-level help to discover accepted path, query, header, and body fields:

```bash
sendmux management:create-domain --help
sendmux mailbox:search-message-snippets --help
sendmux sending:send:batch --help
```

## Operation flags

Operation commands share these flags:

| Flag                  | Use                                                                        |
| --------------------- | -------------------------------------------------------------------------- |
| `--api-key`           | Direct key; overrides profile/env profile lookup.                          |
| `--base-url`          | Override API base URL.                                                     |
| `--profile`, `-p`     | Select a local profile.                                                    |
| `--body`              | Inline JSON request body, or text bytes for byte-oriented operations.      |
| `--body-file`         | Read a JSON request body or byte payload from a file.                      |
| `--path name=value`   | Path parameters. Repeat for multiple path params.                          |
| `--query name=value`  | Query parameters. Repeat for filters and pagination.                       |
| `--header name=value` | Headers accepted by the operation. Repeat for multiple headers.            |
| `--idempotency-key`   | Shortcut for `Idempotency-Key`. Works only when the operation supports it. |
| `--if-match`          | Shortcut for `If-Match`. Works only when the operation supports it.        |
| `--if-none-match`     | Shortcut for `If-None-Match`. Works only when the operation supports it.   |
| `--json`              | Machine-readable output.                                                   |

`--path`, `--query`, and `--header` require `name=value`. Booleans use `true` or `false`. Repeat an array-valued parameter rather than comma-joining it.

Pass either `--body` or `--body-file`, not both.

## Examples

Create a domain:

```bash
sendmux management:create-domain \
  --profile default \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body '{"domain":"example.com","mode":"send_receive"}' \
  --json
```

Get domain DNS records:

```bash
sendmux management:get-domain-zone-file \
  --profile default \
  --path public_id=mdom_abc \
  --json
```

Search a mailbox without reading full messages:

```bash
sendmux mailbox:search-message-snippets \
  --profile mailbox \
  --query q=invoice \
  --query is_unread=true \
  --query limit=10 \
  --json
```

Batch-read selected mailbox messages:

```bash
sendmux mailbox:batch-get-messages \
  --profile mailbox \
  --body '{
    "ids": ["eml_abc", "eml_def"],
    "body_mode": "clean_json",
    "max_body_chars": 4000
  }' \
  --json
```

Send a batch:

```bash
sendmux sending:send:batch \
  --profile sending \
  --idempotency-key "$IDEMPOTENCY_KEY" \
  --body-file ./messages.json \
  --json
```

Poll one unchanged-safe delivery log:

```bash
sendmux management:get-email-log \
  --profile default \
  --path public_id=dlog_abc \
  --if-none-match "$ETAG" \
  --json
```

## Routing

- First setup/auth check: `sendmux-getting-started`.
- Sending strategy and body shape: `sendmux-send-email`.
- Mailbox read, search, sync, triage, or reply: `sendmux-mailbox-agent`.
- Account-level management strategy: `sendmux-management`.
- MCP connection setup: `sendmux-mcp-setup`.
- Cheapest-call doctrine: `sendmux-token-efficient-usage`.

## Limitations

- Does not replace task-specific strategy skills such as sendmux-send-email, sendmux-mailbox-agent, or sendmux-management.
- Does not run destructive commands without explicit confirmation.
- Requires the correct key prefix for the chosen command surface.
