---
name: sendmux-mcp-setup
description: "Configure hosted or local Sendmux MCP servers for mailbox, management, and sending tools."
risk: safe
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-mcp-setup
---

# Sendmux MCP setup

Use this skill to connect an agent client to Sendmux through MCP.

## When to Use This Skill

- Use when connecting an agent client to Sendmux through hosted or local MCP.
- Use when choosing hosted remote OAuth, local stdio, or local HTTP bearer setup.
- Use when writing Sendmux MCP config for Claude Code, Cursor, Codex, VS Code, Copilot, Gemini CLI, Cline, or Windsurf.

## Copy-Paste Example

```text
User: "Configure Sendmux MCP for Claude Code using hosted OAuth if supported; otherwise use local stdio with env vars."
```

## Boundaries

- Do not ask the user to paste API keys or bearer tokens.
- Use `smx_mbx_` keys or scoped `smx_agent_` tokens for Mailbox MCP tools.
- Use send-capable `smx_mbx_` keys or owner-approved Sending-resource `smx_agent_` tokens for Sending MCP tools.
- Use `smx_root_` keys for Management MCP tools.
- Use hosted OAuth at `https://mcp.sendmux.ai/mcp` when the client supports remote MCP OAuth.
- Use local stdio when the client cannot use hosted OAuth or local HTTP.
- For local stdio or HTTP, pass Sendmux keys and owner-approved agent tokens through environment variables backed by the user's secret store; do not write raw tokens into checked-in MCP config.
- Use local HTTP bearer only for local/private MCP servers; the bearer token protects the MCP endpoint and is separate from the Sendmux API key used upstream.
- Use server-qualified names such as `sendmux-mailbox:mailbox_search_message_snippets` when a client needs fully-qualified tool names.

## Install

```bash
pip install sendmux-mcp
```

Console scripts:

- `sendmux-mcp` — combined local server; requires `--surfaces` or `SENDMUX_MCP_SURFACES`.
- `sendmux-mcp-mailbox` — mailbox-only local server.
- `sendmux-mcp-management` — management-only local server.
- `sendmux-mcp-sending` — sending-only local server.
- `sendmux-mcp-hosted` — hosted runtime; do not use this for normal local agent setup.

## Choose A Setup

| Setup             | Use when                                                       | Auth                                                                                  |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Hosted remote     | The client supports remote MCP OAuth.                          | Client signs in through Sendmux OAuth; do not pass API keys.                          |
| Local stdio       | The agent runs a local child process.                          | Env vars passed to the server process.                                                |
| Local HTTP bearer | A local/private MCP endpoint is shared by one or more clients. | Sendmux API key in server env; `Authorization: Bearer ...` from client to MCP server. |

## Surface Map

| Surface    | Key                                                                     | Tool count | Example tools                                                                                                                                         |
| ---------- | ----------------------------------------------------------------------- | ---------: | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mailbox    | `smx_mbx_` or scoped `smx_agent_`                                       |         21 | `mailbox_list_granted_mailboxes`, `mailbox_search_message_snippets`, `mailbox_batch_get_messages`, `mailbox_get_changes`, `mailbox_send_message`      |
| Management | `smx_root_`                                                             |         20 | `management_create_domain`, `management_create_mailbox`, `management_create_mailbox_key`, `management_get_spend_summary`, `management_create_webhook` |
| Sending    | Send-capable `smx_mbx_` or owner-approved Sending-resource `smx_agent_` |          2 | `sending_send_email`, `sending_send_email_batch`                                                                                                      |

For multi-mailbox grants, call `mailbox_list_granted_mailboxes` first and pass the returned `mailbox_id` to mailbox tools when targeting a mailbox.

## Local Servers

Mailbox-only stdio:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux-mcp-mailbox
```

Management-only stdio:

```bash
SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux-mcp-management
```

Sending-only stdio:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux-mcp-sending
```

Combined stdio:

```bash
SENDMUX_MCP_SURFACES=mailbox,management,sending \
SENDMUX_MAILBOX_API_KEY="$SENDMUX_MBX_KEY" \
SENDMUX_MANAGEMENT_API_KEY="$SENDMUX_ROOT_KEY" \
SENDMUX_SENDING_API_KEY="$SENDMUX_MBX_KEY" \
sendmux-mcp
```

Local HTTP bearer:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" \
SENDMUX_MCP_HTTP_BEARER_TOKEN="$SENDMUX_MCP_HTTP_BEARER_TOKEN" \
sendmux-mcp-mailbox --transport http --host 127.0.0.1 --port 8765 --path /mcp
```

Client header for that local HTTP server:

```text
Authorization: Bearer $SENDMUX_MCP_HTTP_BEARER_TOKEN
```

`/health` returns selected surfaces for local HTTP servers.

## Common JSON Clients

Use this shape for Cursor, Cline, and Windsurf/Cascade clients that read an `mcpServers` object.

Local stdio, one mailbox server:

```json
{
  "mcpServers": {
    "sendmux-mailbox": {
      "command": "sendmux-mcp-mailbox",
      "env": {
        "SENDMUX_API_KEY": "${env:SENDMUX_MBX_KEY}"
      }
    }
  }
}
```

Local stdio, all three surfaces:

```json
{
  "mcpServers": {
    "sendmux": {
      "command": "sendmux-mcp",
      "args": ["--surfaces", "mailbox,management,sending"],
      "env": {
        "SENDMUX_MAILBOX_API_KEY": "${env:SENDMUX_MBX_KEY}",
        "SENDMUX_MANAGEMENT_API_KEY": "${env:SENDMUX_ROOT_KEY}",
        "SENDMUX_SENDING_API_KEY": "${env:SENDMUX_MBX_KEY}"
      }
    }
  }
}
```

Local/private HTTP bearer:

```json
{
  "mcpServers": {
    "sendmux-local-http": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${env:SENDMUX_MCP_HTTP_BEARER_TOKEN}"
      }
    }
  }
}
```

Hosted remote OAuth:

```json
{
  "mcpServers": {
    "sendmux": {
      "url": "https://mcp.sendmux.ai/mcp"
    }
  }
}
```

Client notes:

- Cursor: put project config at `.cursor/mcp.json` or global config at `~/.cursor/mcp.json`; Cursor interpolates `${env:NAME}` in `command`, `args`, `env`, `url`, and `headers`.
- Cline: use `~/.cline/mcp.json`, the Cline MCP UI, or `cline mcp`; remote setup can ask for URL and headers.
- Windsurf/Cascade: use `~/.codeium/mcp_config.json` or **Settings** > **Tools** > **Windsurf Settings** > **Add Server**; HTTP config accepts `serverUrl` or `url`.

## Claude Code

Hosted remote OAuth:

```bash
claude mcp add --transport http sendmux https://mcp.sendmux.ai/mcp
```

Then run `/mcp` and complete the sign-in flow if prompted.

Local stdio:

```bash
claude mcp add --transport stdio \
  --env SENDMUX_API_KEY="$SENDMUX_MBX_KEY" \
  sendmux-mailbox -- sendmux-mcp-mailbox
```

Local HTTP bearer:

```bash
claude mcp add --transport http \
  sendmux-local-http http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $SENDMUX_MCP_HTTP_BEARER_TOKEN"
```

Project `.mcp.json` can also use `mcpServers` with `type`, `url` or `command`, `args`, `env`, and `headers`.

## Codex

Use `~/.codex/config.toml` for user-level config.

Local stdio:

```toml
[mcp_servers.sendmux_mailbox]
command = "sendmux-mcp-mailbox"
env_vars = ["SENDMUX_API_KEY"]
```

Run Codex with `SENDMUX_API_KEY` set to an `smx_mbx_` key.

Combined stdio:

```toml
[mcp_servers.sendmux]
command = "sendmux-mcp"
args = ["--surfaces", "mailbox,management,sending"]
env_vars = ["SENDMUX_MAILBOX_API_KEY", "SENDMUX_MANAGEMENT_API_KEY", "SENDMUX_SENDING_API_KEY"]
```

Hosted remote OAuth:

```toml
[mcp_servers.sendmux]
url = "https://mcp.sendmux.ai/mcp"
oauth_resource = "https://mcp.sendmux.ai/mcp"
```

Local HTTP bearer:

```toml
[mcp_servers.sendmux_local_http]
url = "http://127.0.0.1:8765/mcp"
bearer_token_env_var = "SENDMUX_MCP_HTTP_BEARER_TOKEN"
```

## VS Code And GitHub Copilot

VS Code stores MCP config in `.vscode/mcp.json` or user profile `mcp.json` under `servers`.

Local stdio:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "sendmux-mbx-key",
      "description": "Sendmux mailbox API key",
      "password": true
    }
  ],
  "servers": {
    "sendmuxMailbox": {
      "type": "stdio",
      "command": "sendmux-mcp-mailbox",
      "env": {
        "SENDMUX_API_KEY": "${input:sendmux-mbx-key}"
      }
    }
  }
}
```

Local HTTP bearer:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "sendmux-mcp-token",
      "description": "Sendmux local MCP bearer token",
      "password": true
    }
  ],
  "servers": {
    "sendmuxLocalHttp": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${input:sendmux-mcp-token}"
      }
    }
  }
}
```

Hosted remote OAuth:

```json
{
  "servers": {
    "sendmux": {
      "type": "http",
      "url": "https://mcp.sendmux.ai/mcp"
    }
  }
}
```

GitHub Copilot CLI can add servers interactively with `/mcp add` or non-interactively:

```bash
copilot mcp add sendmux-mailbox -- sendmux-mcp-mailbox
copilot mcp add --transport http sendmux https://mcp.sendmux.ai/mcp
copilot mcp add --transport http sendmux-local-http http://127.0.0.1:8765/mcp \
  --header "Authorization: Bearer $SENDMUX_MCP_HTTP_BEARER_TOKEN"
```

## Gemini CLI

Gemini CLI reads `mcpServers` from `settings.json`.

Local stdio:

```json
{
  "mcpServers": {
    "sendmux-mailbox": {
      "command": "sendmux-mcp-mailbox",
      "env": {
        "SENDMUX_API_KEY": "$SENDMUX_MBX_KEY"
      },
      "trust": false
    }
  }
}
```

Hosted remote OAuth:

```json
{
  "mcpServers": {
    "sendmux": {
      "httpUrl": "https://mcp.sendmux.ai/mcp",
      "trust": false
    }
  }
}
```

Local HTTP bearer:

```json
{
  "mcpServers": {
    "sendmux-local-http": {
      "httpUrl": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer $SENDMUX_MCP_HTTP_BEARER_TOKEN"
      },
      "trust": false
    }
  }
}
```

Use `/mcp auth sendmux` if the hosted remote endpoint needs OAuth authentication.

## Verification

After adding the server:

1. Restart or refresh MCP servers in the client.
2. Confirm the visible tools match the selected surfaces:
   - Mailbox-only: no `management_*` or `sending_*` tools.
   - Management-only: no `mailbox_*` or `sending_*` tools.
   - Sending-only: only `sending_send_email` and `sending_send_email_batch`.
3. Run one harmless read tool:
   - Mailbox: `mailbox_get_me` or `mailbox_get_session`.
   - Management: `management_list_domains` with a small limit.
   - Sending: list tools only unless the user confirms a real send.
4. If local HTTP returns `401`, check the client `Authorization` header against `SENDMUX_MCP_HTTP_BEARER_TOKEN`.
5. If the process exits before connecting, check the Sendmux key prefix for the selected surface.

## Routing

- First Sendmux API setup or first call: `sendmux-getting-started`.
- Sending body shape or send strategy: `sendmux-send-email`.
- Mailbox read, search, sync, triage, or reply: `sendmux-mailbox-agent`.
- Account-level management strategy: `sendmux-management`.
- Terminal command mechanics: `sendmux-cli`.
- Cheapest-call doctrine: `sendmux-token-efficient-usage`.

## Limitations

- Does not ask for or write raw API keys into checked-in MCP config.
- Does not send email; the source says list Sending tools only unless the user confirms a real send.
- Requires client-specific MCP support for hosted OAuth, stdio, or HTTP bearer transport.
