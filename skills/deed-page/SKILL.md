---
name: deed-page
description: "Publish a generated HTML file or static folder to a public https://<slug>.deed.page/ URL after the user approves a preview, then update or delete it."
risk: critical
source: https://github.com/f-o-x11/deed-page-skill
---

# deed.page publish

## Overview

[deed.page](https://deed.page) hosts static files for agents. One `curl` upload of an HTML file or a tarred folder returns a live `https://<slug>.deed.page/` URL. No account, API key, or email code is needed; the first publish returns an owner token that is the only key for later updates and deletes.

Risk: **critical**. Publishing makes files public on the internet, and update/delete change a live site. This skill never publishes on its own: the agent shows the user a preview and waits for an explicit "yes" before every upload, update, or delete.

## When to Use This Skill

- Use when the user asks to publish, host, share online, "put this online", or "give me a link" for something you generated: an HTML report, dashboard, slide deck, landing page, prototype, game, or docs folder.
- Use when the user wants to update or take down a deed.page site they published earlier.
- Do not use for secrets, private data, or anything needing server-side code or a database.
- Do not use in unattended or background runs where no human can approve the preview. Build the files locally and stop.

## Step-by-Step Guide

### 1. Load the owner token without showing it

Keep one token per agent in private durable storage, for example `~/.config/deed-page/token` (`chmod 600` on the file, `chmod 700` on the directory), or in the `DEEDPAGE_TOKEN` environment variable. The token is a credential: never `echo` or `cat` it to the terminal, never paste it into the chat, and never put it in a page, commit, or log. Load it straight into a header argument:

```sh
TOKEN=${DEEDPAGE_TOKEN:-$(cat ~/.config/deed-page/token 2>/dev/null)}
set --
if [ -n "$TOKEN" ]; then set -- -H "Authorization: Bearer $TOKEN"; fi
```

If no token is saved yet, the first publish creates one (step 5). If the host has no private persistent storage, tell the user and offer to keep the token in a credential store they manage.

### 2. Pick a slug

Lowercase letters, digits, and hyphens, 1 to 63 characters, no leading or trailing hyphen. `hello`, `my-page`, `my-app`, and `my-site` are reserved. Add a random suffix to avoid collisions:

```bash
SLUG="report-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
```

### 3. Show a preview and ask for an explicit yes

Before any upload, show the user exactly what will become public and stop until they answer:

```sh
echo "Destination: https://$SLUG.deed.page/ (public, anyone with the link can view it)"
find ./site -type f | sort        # or: ls -l report.html
```

Ask: "Publish these files to https://<slug>.deed.page/? (yes/no)". Continue only on a clear yes. If the list contains `.env` files, keys, credentials, personal data, or a whole project checkout, point that out and publish only a clean output folder such as `dist/`. A yes covers this one publish; ask again for the next update or delete.

### 4. Publish (only after the yes)

Single HTML file (stored as `index.html`):

```sh
RESP=$(curl -sS -T report.html "https://deed.page/v1/sites?slug=$SLUG" -H "X-Client: <your agent name>" "$@")
```

Folder (must contain `index.html` at its root):

```sh
tar -C ./site -czf /tmp/site.tar.gz .
RESP=$(curl -sS -T /tmp/site.tar.gz "https://deed.page/v1/sites?slug=$SLUG" -H "X-Client: <your agent name>" "$@")
```

`"$@"` carries the `Authorization` header from step 1, so the token never appears in the command text.

### 5. Save a new token privately and answer the user

A first publish without a token returns `token` once. Write it to the private file without printing it, then show the response with the token masked. The file is written only when the response has a non-empty token, so a saved token is never replaced with an empty one:

```sh
NEW=$(printf '%s' "$RESP" | sed -n 's/.*"token": *"\(deed_tok_[^"]*\)".*/\1/p')
if [ -n "$NEW" ]; then
  (umask 077 && mkdir -p ~/.config/deed-page && chmod 700 ~/.config/deed-page && printf '%s' "$NEW" > ~/.config/deed-page/token && chmod 600 ~/.config/deed-page/token)
  TOKEN=$NEW; set -- -H "Authorization: Bearer $TOKEN"; unset NEW
fi
printf '%s\n' "$RESP" | sed 's/"token": *"[^"]*"/"token":"<saved>"/'
```

Reply with the `url` field and say where the token is stored, without showing its value.

### 6. Update or delete

To ship a new version, repeat steps 3 and 4 for the same slug; the saved token is sent through `"$@"`. Delete only when the user asks for it and confirms the slug:

```bash
curl -sS -X DELETE "https://deed.page/v1/sites/$SLUG" "$@"
```

## Examples

### Example 1: Share a generated report

```sh
SLUG="q3-report-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
TOKEN=${DEEDPAGE_TOKEN:-$(cat ~/.config/deed-page/token 2>/dev/null)}
set --
if [ -n "$TOKEN" ]; then set -- -H "Authorization: Bearer $TOKEN"; fi
echo "Will publish q3-report.html to https://$SLUG.deed.page/ (public)"
# Ask the user "Publish? (yes/no)" and stop here until they say yes.
RESP=$(curl -sS -T q3-report.html "https://deed.page/v1/sites?slug=$SLUG" -H "X-Client: Claude Code" "$@")
# {"ok":true,"url":"https://q3-report-1a2b3c4d.deed.page/","slug":"q3-report-1a2b3c4d","version":"v1","token":"deed_tok_..."}
# Save the token and print the masked response as in step 5.
```

### Example 2: Inline page with no file on disk

Show the user the HTML you are about to publish and the target URL first, then:

```sh
RESP=$(curl -sS -X POST "https://deed.page/v1/sites" -H 'content-type: application/json' -H "X-Client: Codex" "$@" \
  -d "{\"slug\":\"$SLUG\",\"files\":{\"index.html\":\"<!doctype html><h1>Hello</h1>\"}}")
```

## Best Practices

- ✅ **Do:** show the public URL and the exact file list, and wait for an explicit yes before every publish, update, or delete.
- ✅ **Do:** keep the token in a `chmod 600` file or `DEEDPAGE_TOKEN` and pass it through `"$@"`; mask it in any output.
- ✅ **Do:** add `-H "X-Ttl: 86400"` for throwaway previews that should expire.
- ❌ **Don't:** publish from unattended runs, or treat an earlier yes as approval for a new upload.
- ❌ **Don't:** print, log, or paste the owner token, or publish credentials, `.env` files, or personal data.
- ❌ **Don't:** delete or overwrite a site the user did not ask you to change.

## Limitations

- Static files only: no server-side code, databases, or build step.
- Free plan: 25 sites, 25 MB per site, 60 publishes per hour, 4 MB per single request (larger uploads use the manifest flow in the docs).
- Free sites show a small "Published with deed.page" badge; paid plans remove it.
- A lost token cannot be recovered unless a recovery email was attached; the site stays online.

## Troubleshooting

**Problem:** `409 slug_taken`
**Solution:** Use the `suggestion` field from the response or a new random suffix, and show the user the new URL before retrying.

**Problem:** `422 content_rejected`
**Solution:** The upload contained executable or phishing markers. Remove that file; do not retry as-is.

**Problem:** `403` or `413` plan-limit error, or `429 rate_limited`
**Solution:** Show the response's `message_for_user` to the user as-is and do not retry the same request (for 429, not before `reset_at`).

Full API, MCP server (`https://deed.page/mcp`), versions, custom domains, and passwords: https://deed.page/docs
