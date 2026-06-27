---
name: sendmux-getting-started
description: "Set up Sendmux, choose key scopes, verify first harmless calls, and route agent registration."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-getting-started
---

# Sendmux getting started

Use this skill to get a user from "I have a Sendmux task" to the correct surface, key kind, package, and first verified call.

## When to Use This Skill

- Use when setting up Sendmux tools, skills, keys, or first verification calls.
- Use when choosing between MCP, CLI, SDK, or direct HTTP for a Sendmux task.
- Use when an agent needs Sendmux self-registration, owner invite, or credential-bundle handling.

## Copy-Paste Example

```text
User: "Set up Sendmux for this agent and verify the mailbox key without sending email."
```

## Safety first

- Do not ask the user to paste an API key.
- Do not print API keys.
- Prefer existing environment variables, local CLI profiles, or the user's secret manager.
- Store the self-registration credential bundle in a secure store immediately after registration; later MCP, CLI, SDK, mailbox, and sending work depends on it.
- Never rely on chat, logs, repo files, screenshots, or memory-only state for `claim_token` or agent credentials.
- Keep the `identity_assertion` or pre-claim `smx_agent_*` token available until the owner invite returns `202`; `claim_token` is only for post-approval exchange.
- If a key appears in chat or logs, stop and tell the user to rotate it before continuing.

## Install Sendmux skills

If the agent supports Skills and the Sendmux skills are not installed, install the pack first:

```bash
npx skills add Sendmux/skills
```

Skills are optional. If the agent cannot install skills, continue from `https://app.sendmux.ai/auth.md`.

## Pick the key

| Task                                                                                    | Key prefix                                                              | Start here                                                                                                               |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Send email through the Sending API                                                      | Send-capable `smx_mbx_` or owner-approved Sending-resource `smx_agent_` | `sendmux-send-email` for real sends; this skill can verify package/API discovery first.                                  |
| Read, search, sync, triage, or reply from one mailbox                                   | `smx_mbx_` or scoped `smx_agent_`                                       | Mailbox MCP, CLI, or SDK.                                                                                                |
| Manage domains, mailboxes, mailbox keys, providers, webhooks, logs, billing, or metrics | `smx_root_`                                                             | Management MCP, CLI, or SDK.                                                                                             |
| Let an agent register itself and invite its owner                                       | No existing key, then `smx_agent_`                                      | Agent access: `/auth.md`, challenge endpoint, `identity_endpoint`, token endpoint, `/api/v1/mailbox/me`, and invite endpoint. |

If the task mixes management and mailbox work, use separate keys and separate clients or profiles. Do not use a root key for mailbox-scoped examples.

## Choose the surface

1. Use MCP first when the user's agent already has the relevant `sendmux-mcp` server connected and the needed tool is curated.
2. Use the `sendmux` CLI for one-shot terminal work, debugging, shell scripts, and examples the user can copy into a terminal. Add `--json` so downstream agents can parse the envelope.
3. Use an SDK when writing application code. Install only the package for the chosen surface unless the project needs multiple surfaces.
4. Use direct HTTP only when the user's environment cannot use MCP, CLI, or an SDK.

## Install the relevant package

CLI:

```bash
npm install -g @sendmux/cli
```

MCP:

```bash
pipx install sendmux-mcp
```

TypeScript SDK packages:

```bash
npm install @sendmux/mailbox
npm install @sendmux/management
npm install @sendmux/sending
```

Use the one package matching the task; do not install all three unless the project needs all three.

## First verified calls

### Mailbox key, mailbox work

MCP tool:

```text
mailbox_get_me
```

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux profiles:set mailbox --default --json
sendmux mailbox:me:get --json
```

SDK:

```ts
import { createMailboxClient, mailboxGetMe } from "@sendmux/mailbox";

const client = createMailboxClient({ apiKey: process.env.SENDMUX_API_KEY! });
const response = await mailboxGetMe({ client });
console.log(response.data);
```

This call resolves the mailbox behind the bearer token and should be the default harmless first call for `smx_mbx_` and scoped `smx_agent_` mailbox workflows.

### Self-registered agent token

Use this when the agent has no human-created key yet.

1. Read `https://app.sendmux.ai/auth.md`.
2. Use the copy-paste proof helper in `/auth.md` if the agent needs exact solver code.
3. Request a registration challenge with the intended anonymous registration body at `POST /agent-auth/agent/identity/challenge`.
4. Solve the returned proof-of-work challenge.
5. Encode `proof_of_work` as base64 UTF-8 JSON with `{ "challenge": <exact challenge>, "solution": { "counter": <integer>, "derivedKey": <hex> } }`.
6. Create an anonymous identity with the same body plus `proof_of_work` at `identity_endpoint`, `POST /agent-auth/agent/identity`.
7. Immediately save the returned `claim_token` and agent credential bundle in a secure store before token exchange, then exchange the returned `identity_assertion` with `POST /agent-auth/oauth2/token`.
8. Call `GET /api/v1/mailbox/me` with the returned `smx_agent_` token.
9. Keep the `identity_assertion` or pre-claim token available until the owner invite returns `202`; `claim_token` cannot create the invite.
10. Do not stop after the pre-claim token works. If the owner email is known, request the owner invite with `POST /agent-auth/agent/identity/invite`. If it is unknown, ask: `What owner email should I invite for approval?`
11. After the owner accepts and approves sending in Sendmux, exchange `claim_token` with the claim grant; request `resource=https://smtp.sendmux.ai/api/v1` before Sending API calls.

Pre-claim `smx_agent_` tokens have `mailbox.read` and `email.receive`. They do not have `email.send`; owner-approved Sending-resource `smx_agent_` tokens can send from the assigned mailbox. Sendmux sends the owner invite through the invite endpoint. Only one live pre-claim owner invite can be pending; retry the same request with the same idempotency key. If token exchange returns expected `503` states, wait for `Retry-After` or `retry_after`. If registration returns `503 server_error`, stop and report it instead of looping. If agent auth returns `429`, wait for `Retry-After` or `retry_after` before retrying.

The raw `claim_token` is shown once and is required after owner approval. Sendmux cannot recover it later. Immediately after registration, store the agent credential bundle: `claim_token`, `registration_id`, `mailbox.email`, `claim_token_expires`, `identity_assertion`, `token_endpoint`, the app resource URL, and the sending resource URL. Use a secure store such as 1Password, an OS keychain, or the agent platform's encrypted secret store. After token exchange, add the pre-claim token or keep `identity_assertion` available until invite `202`. If no secure store is available, stop and ask the user where to store the bundle before sending the owner invite. If the claim token was lost, rerun registration and invite with a fresh agent identity. If the `identity_assertion` and pre-claim token were lost before the invite returned `202`, rerun registration and invite immediately in the same flow.

### Root key, management work

MCP tool:

```text
management_list_mailboxes
```

CLI:

```bash
SENDMUX_API_KEY="$SENDMUX_ROOT_KEY" sendmux profiles:set root --default --json
sendmux management:mailboxes:list --query limit=1 --json
```

SDK:

```ts
import {
  createManagementClient,
  managementListMailboxes,
} from "@sendmux/management";

const client = createManagementClient({ apiKey: process.env.SENDMUX_API_KEY! });
const response = await managementListMailboxes({
  client,
  query: { limit: 1 },
});
console.log(response.data);
```

Use a small list call as the first management check. It verifies the root key and avoids creating or changing resources.

### Sending work

The Sending surface needs a send-capable `smx_mbx_` key with `email.send` or an owner-approved Sending-resource `smx_agent_` token. Do not send a real email as a health check unless the user explicitly asks to send one and provides the message details.

CLI package/API discovery:

```bash
SENDMUX_API_KEY="$SENDMUX_MBX_KEY" sendmux sending:get-open-api-spec --json
```

SDK package/API discovery:

```ts
import { createSendingClient, sendingGetOpenApiSpec } from "@sendmux/sending";

const client = createSendingClient({ apiKey: process.env.SENDMUX_API_KEY! });
const response = await sendingGetOpenApiSpec({ client });
console.log(response.data.info);
```

For a real send, route to `sendmux-send-email` and include an `Idempotency-Key`.

## Interpret failures

- Prefix error: the selected surface and credential do not match. Switch to `smx_mbx_` or scoped `smx_agent_` for Mailbox, a send-capable `smx_mbx_` key or owner-approved Sending-resource `smx_agent_` token for Sending, or `smx_root_` for Management.
- `401`: key missing, invalid, or revoked.
- `403`: key is valid but lacks the permission or surface required by the call.
- `429` or `503`: retry according to the response headers; do not loop manually.
- Empty list with `ok: true`: auth worked; there may be no resources yet.

## Route after setup

- Sending one or many outbound messages: `sendmux-send-email`.
- Reading, searching, syncing, or replying from a mailbox: `sendmux-mailbox-agent`.
- Managing domains, mailboxes, keys, webhooks, spend, logs, or metrics: `sendmux-management`.
- CLI-specific workflows: `sendmux-cli`.
- MCP client configuration: `sendmux-mcp-setup`.
- Choosing the cheapest call pattern: `sendmux-token-efficient-usage`.

## Limitations

- Does not replace the live Sendmux auth discovery document at https://app.sendmux.ai/auth.md.
- Cannot recover a lost claim_token; the source says rerun registration with a fresh agent identity.
- Does not send real email as a health check unless the user explicitly asks and provides message details.
