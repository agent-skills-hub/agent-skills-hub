---
name: sendmux-email-for-agents
description: "Route agent email workflows across Sendmux setup, self-registration, mailbox runtime, sending, and approval."
risk: critical
source: https://github.com/Sendmux/skills/tree/main/skills/sendmux-email-for-agents
---

# Sendmux email for agents

Use this skill when the user describes the agent-email problem: an AI agent needs an inbox, mailbox identity, outbound email, triage loop, reply workflow, or human approval path.

## When to Use This Skill

- Use when an AI agent needs its own inbox, mailbox identity, outbound email, triage loop, reply workflow, or human approval path.
- Use when routing a Sendmux agent-email task across management, MCP setup, mailbox runtime, sending, CLI, or token-efficient usage skills.
- Use even when the user describes the agent-email problem without naming Sendmux.

## Copy-Paste Example

```text
User: "Give my support agent its own Sendmux inbox, connect it through MCP, and require approval before replies are sent."
```

## First route

| User problem                               | Route                                                                                                                                                       |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Give my agent an email address"           | `sendmux-management` to create/inspect domain, mailbox, and mailbox key.                                                                                    |
| "Let my agent register itself"             | Agent access: read `/auth.md`, optionally install skills with `npx skills add Sendmux/skills`, register at `identity_endpoint` with `proof_of_work`, immediately save the returned `claim_token` and agent credential bundle securely before token exchange, exchange the assertion, keep the assertion or pre-claim token until invite `202`, verify `/api/v1/mailbox/me`, then invite the owner or ask for the owner email. |
| "Connect my agent to its inbox"            | `sendmux-mcp-setup` for agent MCP, or `sendmux-getting-started` for first auth checks.                                                                      |
| "Read, search, triage, label, sync, reply" | `sendmux-mailbox-agent` with an `smx_mbx_*` key or scoped `smx_agent_*` token.                                                                              |
| "Send independent outbound notifications"  | `sendmux-send-email` with a send-capable `smx_mbx_*` key or owner-approved Sending-resource `smx_agent_*` token; batch when there is more than one message. |
| "Build this into an app or worker"         | SDK path from the task skill; use `sendmux-token-efficient-usage` for call minimisation.                                                                    |
| "Show terminal commands"                   | `sendmux-cli`.                                                                                                                                              |

If the task crosses setup and runtime, split it:

1. `sendmux-management` provisions the mailbox and mailbox API key with an `smx_root_*` key.
2. Runtime agent work uses the new `smx_mbx_*` key.
3. `sendmux-mcp-setup` connects the agent if the client supports MCP.
4. `sendmux-mailbox-agent` handles ongoing mailbox read/triage/reply.

For self-registration without a human-created key, use agent access instead:

1. Read `https://app.sendmux.ai/auth.md`.
2. Use the copy-paste proof helper in `/auth.md` if the agent needs exact solver code.
3. Send the intended anonymous registration body to `POST /agent-auth/agent/identity/challenge`.
4. Solve the returned proof-of-work challenge.
5. Encode `proof_of_work` as base64 UTF-8 JSON with `{ "challenge": <exact challenge>, "solution": { "counter": <integer>, "derivedKey": <hex> } }`.
6. Create the anonymous identity with the same body plus `proof_of_work` at `identity_endpoint`, `POST /agent-auth/agent/identity`.
7. Immediately save the returned `claim_token` and agent credential bundle in a secure store before token exchange, then exchange the returned `identity_assertion` at `POST /agent-auth/oauth2/token`.
8. Call `GET /api/v1/mailbox/me` with the returned pre-claim `smx_agent_*` token.
9. Keep the `identity_assertion` or pre-claim token available until the owner invite returns `202`; `claim_token` cannot create the invite.
10. Do not stop after the pre-claim token works. If the owner email is known, request the owner invite with `POST /agent-auth/agent/identity/invite`. If it is unknown, ask: `What owner email should I invite for approval?`
11. After the owner accepts and approves sending in Sendmux, exchange `claim_token` with the claim grant for an app-resource or Sending-resource `smx_agent_*` token.

## Safety boundaries

- Do not ask the user to paste API keys, mailbox keys, OAuth tokens, or one-time secrets.
- Store the self-registration credential bundle in a secure store immediately after registration; later MCP, CLI, SDK, mailbox, and sending work depends on it.
- Never rely on chat, logs, repo files, screenshots, or memory-only state for `claim_token` or agent credentials.
- Keep the `identity_assertion` or pre-claim `smx_agent_*` token available until the owner invite returns `202`. If both are lost before the invite succeeds, register a fresh identity and invite immediately.
- Do not poll the claim-token grant before the owner invite returns `202`; after `202`, polling with `claim_token` is the wait-for-approval path. `claim_token` cannot create the owner invite.
- Do not send email until the user has supplied or confirmed the recipient, subject, body, and attachments.
- Treat "draft for approval" as a draft. Ask for explicit approval before calling `mailbox_send_message`, `sending_send_email`, or `sending_send_email_batch`.
- Use separate scopes: `smx_root_*` for provisioning/admin, send-capable `smx_mbx_*` keys or owner-approved Sending-resource `smx_agent_*` tokens for Sending, `smx_mbx_*` keys for normal Mailbox runtime, and `smx_agent_*` only for the scopes it was issued with.
- Do not use a root key inside an agent that only needs mailbox read/send work.
- Pre-claim `smx_agent_*` tokens include `mailbox.read` and `email.receive`, not `email.send`.
- Owner invites are sent by Sendmux through the invite endpoint. Do not route them through the Sending API.
- Only one live pre-claim owner invite can be pending; retry the same request with the same idempotency key.
- Token exchange can return expected `503` states while mailbox provisioning or owner approval is pending. Wait for `Retry-After` or `retry_after`. Registration `503 server_error` means stop and report, not repeated retries.
- Agent-auth `429` responses include retry timing; wait for `Retry-After` or `retry_after` before trying again.
- Confirm destructive mailbox actions before delete, permanent delete, key revocation, suspend, or resume.

## Workflow patterns

### New agent inbox

Use when the user wants a new mailbox for an agent, such as support intake, invoice triage, scheduling, approvals, or lead qualification.

Plan:

1. Domain and mailbox setup: route to `sendmux-management`.
2. Mailbox key: create a mailbox-scoped key for the agent runtime.
3. Connection: route to `sendmux-mcp-setup` if the agent client can use MCP; otherwise use CLI or SDK.
4. First harmless check: `mailbox_get_me`, CLI `mailbox:me:get`, or SDK `mailboxGetMe`.
5. Runtime loop: route read/search/sync/reply tasks to `sendmux-mailbox-agent`.

Mention that DNS/domain setup may be required before a custom address receives mail.

### Self-registered agent inbox

Use when the user wants the agent to start without a human-created API key.

Plan:

1. Read discovery from `https://app.sendmux.ai/auth.md`.
2. Use the copy-paste proof helper in `/auth.md` if exact solver code is needed.
3. Request a registration challenge at `/agent-auth/agent/identity/challenge` with the intended anonymous registration body.
4. Solve the returned challenge.
5. Encode `proof_of_work` as base64 UTF-8 JSON with `{ "challenge": <exact challenge>, "solution": { "counter": <integer>, "derivedKey": <hex> } }`.
6. Create the anonymous identity at `identity_endpoint`, `/agent-auth/agent/identity`, with the same body plus `proof_of_work`.
7. Immediately save the returned `claim_token` and agent credential bundle in a secure store before token exchange, then exchange `identity_assertion` at `/agent-auth/oauth2/token`.
8. Call `/api/v1/mailbox/me` with the pre-claim `smx_agent_*` token.
9. Keep the `identity_assertion` or pre-claim token available until the owner invite returns `202`; `claim_token` cannot create the invite.
10. Do not stop after the pre-claim token works. If the owner email is known, request an owner invite at `/agent-auth/agent/identity/invite`. If it is unknown, ask: `What owner email should I invite for approval?`
11. After owner approval, exchange `claim_token` at `/agent-auth/oauth2/token` with Sendmux's documented claim grant; request `resource=https://smtp.sendmux.ai/api/v1` before Sending API calls.

Do not say the pre-claim agent can send email. It cannot. After owner approval, a Sending-resource claim-grant `smx_agent_*` token can send from the assigned mailbox. Sendmux sends the owner invite email separately. Only one live pre-claim owner invite can be pending; retry the same request with the same idempotency key. On expected token-exchange `503` states, wait for `Retry-After` or `retry_after`. On registration `503 server_error`, stop and report the failure. On `429`, wait for `Retry-After` or `retry_after`.

The raw `claim_token` is shown once and is required after owner approval. Sendmux cannot recover it later. Immediately after registration, store the agent credential bundle: `claim_token`, `registration_id`, `mailbox.email`, `claim_token_expires`, `identity_assertion`, `token_endpoint`, the app resource URL, and the sending resource URL. Use a secure store such as 1Password, an OS keychain, or the agent platform's encrypted secret store. After token exchange, add the pre-claim token or keep `identity_assertion` available until invite `202`. If no secure store is available, stop and ask the user where to store the bundle before sending the owner invite. If the claim token was lost, rerun registration and invite with a fresh agent identity. If the `identity_assertion` and pre-claim token were lost before the invite returned `202`, rerun registration and invite immediately in the same flow.

### Agent triage loop

Use mailbox-efficient calls:

1. `mailbox_get_changes` or query-change endpoints to resume from the prior state.
2. `mailbox_count_messages` for counts.
3. `mailbox_search_message_snippets` with small `limit` for candidate messages.
4. `mailbox_batch_get_messages` for selected IDs.
5. `mailbox_batch_update_messages` only after the user confirms labels, flags, or read-state changes.

Store state tokens. Do not rescan the whole mailbox.

### Human-approved replies

Use when the agent should prepare a reply but a person approves the send.

Plan:

1. Use `sendmux-mailbox-agent` to read the relevant message or thread.
2. Produce the draft text and list the target message/thread.
3. Ask for approval with the exact recipient, subject, and body.
4. After approval, send with `mailbox_send_message` for mailbox-centred replies.
5. Use `Idempotency-Key` for retryable sends.

### Outbound notifications

Use `sendmux-send-email` when the email is not a reply inside an active mailbox workflow.

- One message: `sending_send_email`, CLI `sending:send`, or SDK `sendingSendEmail`.
- More than one message: `sending_send_email_batch`, CLI `sending:send:batch`, or SDK `sendingSendEmailBatch`.
- Use `Idempotency-Key` and inspect per-message batch results.

## Output shape

When designing a workflow, answer in this order:

1. **Recommended route:** name the Sendmux skill(s) to use next.
2. **Key scope:** `smx_root_*` for admin, `smx_mbx_*` for normal runtime, `smx_agent_*` for scoped self-registered agent runtime, and owner-approved Sending-resource `smx_agent_*` for agent sending.
3. **Runtime surface:** MCP when curated and connected, CLI for terminal work, SDK for application code.
4. **Core calls:** list the smallest Sendmux calls needed.
5. **Human approval:** state what must be confirmed before sending or mutating mail.
6. **Efficiency:** name the batch, snippet, count, delta, cursor, ETag, or idempotency pattern that avoids extra work.

## Do not over-answer

This is a router and architecture skill. Hand detailed implementation to the task skill once the route is clear:

- `sendmux-management`: domains, mailbox provisioning, mailbox keys, account admin, webhooks, billing, logs.
- `sendmux-mailbox-agent`: mailbox read/search/sync/triage/reply.
- `sendmux-send-email`: send bodies, attachments, batch send, HTTP-vs-SMTP send choice.
- `sendmux-mcp-setup`: client configuration and hosted/local MCP.
- `sendmux-cli`: exact terminal commands and flags.
- `sendmux-token-efficient-usage`: cheapest-call doctrine across surfaces.

## Limitations

- This is a router and architecture skill; detailed implementation belongs in the task-specific Sendmux skills named by the source.
- Does not send email or mutate mail without explicit human approval.
- Cannot recover a lost claim_token; the source says rerun registration and invite with a fresh agent identity.
