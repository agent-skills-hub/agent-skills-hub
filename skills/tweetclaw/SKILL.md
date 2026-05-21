---
name: tweetclaw
description: "Use TweetClaw for safety-reviewed X/Twitter automation from agent workflows: search tweets, post tweets, manage replies, export followers, monitor accounts, run giveaway draws, and call Xquik from OpenClaw."
risk: safe
source: https://github.com/Xquik-dev/tweetclaw
---

# TweetClaw

TweetClaw is the Xquik OpenClaw plugin for structured X/Twitter automation. Use it when an agent needs API-backed X/Twitter workflows with clear user approval, secret handling, and bounded scope.

## When to Use This Skill

- Use when the user asks to scrape tweets, search tweets, or search tweet replies.
- Use when the user asks to post tweets, post tweet replies, upload media, or download tweet media.
- Use when the user asks for follower export, user lookup, direct messages, X trends, or tweet monitors.
- Use when the user wants giveaway draws from replies, likes, retweets, or followers.
- Use when an OpenClaw workflow needs the `@xquik/tweetclaw` plugin or Xquik API setup guidance.

## When Not to Use This Skill

- Do not use it for spam, deceptive engagement, impersonation, harassment, or bulk unsolicited DMs.
- Do not use it to bypass platform rules, collect credentials, or hide automation from users.
- Do not perform writes, paid calls, recurring monitors, webhooks, or bulk exports without explicit user approval.
- Do not print API keys, signing keys, cookies, account tokens, or private direct messages unless the user explicitly asks for specific account-owned data.

## Setup

Install the OpenClaw plugin:

```bash
openclaw plugins install npm:@xquik/tweetclaw
```

Use an Xquik API key from the dashboard for account-backed workflows. Store the key in OpenClaw plugin config or an environment variable, not in chat or committed files.

If the plugin is installed without credentials, use the free discovery mode first. Live calls should return setup guidance until a valid key is configured.

## Step-by-Step Guide

1. Identify the exact X/Twitter job: read, write, export, monitor, webhook, or draw.
2. Confirm the user owns or is authorized to access the target account or data.
3. Use TweetClaw discovery to find the matching endpoint and required parameters.
4. For reads, keep limits narrow by default and state what data will be returned.
5. For writes, show the final text, target account, reply target, links, and media before sending.
6. For paid, bulk, recurring, or visible actions, state the scope and wait for explicit confirmation.
7. Invoke the TweetClaw endpoint only after the user approves the action.
8. Summarize the result with IDs, counts, monitor names, or webhook destinations as relevant.

## Common Workflows

### Search Tweets

Use for recent posts, reply research, trend checks, and topic monitoring.

```text
User: Search tweets about OpenClaw plugins from the last 24 hours.
Agent: Confirms scope, uses TweetClaw search, returns matched tweets with IDs, authors, timestamps, and URLs.
```

### Post a Tweet or Reply

Use only after explicit approval.

```text
User: Post this launch note from my connected account.
Agent: Shows the final post text, account, and media list, then waits for approval before calling TweetClaw.
```

### Export Followers

Use for account-owned audience research, CRM enrichment, or giveaway verification.

```text
User: Export 100 followers for @example.
Agent: Confirms authorization and limit, runs the export, then returns a count and output location or structured rows.
```

### Run a Giveaway Draw

Use for transparent winner selection from tweet engagement.

```text
User: Pick 3 winners from replies to this tweet.
Agent: Confirms tweet URL, entry rules, exclusions, and winner count, then runs the draw and reports selected entries.
```

## Best Practices

- Keep default limits small until the user asks for a larger extraction.
- Prefer structured IDs and URLs in summaries so the user can audit results.
- Redact secrets, account credentials, private tokens, and unrelated personal data.
- Ask for a second confirmation when the user changes scope after approving a write, draw, monitor, webhook, or export.
- Use the canonical docs for current setup and endpoint details: https://docs.xquik.com
