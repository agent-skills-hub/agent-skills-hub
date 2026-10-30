---
name: curlshot
description: "Capture web pages as PNG/JPEG/WebP screenshots, PDFs or scrolling videos, or render HTML/Markdown, with the CurlShot screenshot API and curl."
risk: safe
source: https://github.com/SergiuPogor/curlshot-mcp/tree/main/skills/curlshot
license: MIT
---

# CurlShot screenshots

## Overview

CurlShot renders a web page in a real browser and returns it as an image (PNG, JPEG, WebP), a PDF, or a video of the page scrolling (MP4, WebM, GIF). One HTTP request does it: `GET https://curlshot.com/api/v1/screenshot?url=…`, or a POST with a JSON body for your own HTML or Markdown.

## When to Use This Skill

- Use when the user wants a screenshot or a full-page capture of a public URL.
- Use when the user wants to see a page as it looks on a phone, a tablet, or in dark mode.
- Use when the user wants a web page saved as a PDF, or as a scrolling video (MP4, WebM, GIF).
- Use when the user wants their own HTML or Markdown (a social card, a report, a local page) rendered to an image or PDF.

## Prerequisites

- `curl` on the machine and network access to `https://curlshot.com`.
- A CurlShot access key in `CURLSHOT_API_KEY`. If it is not set, ask the user for it. A free key comes from https://curlshot.com/register: 100 screenshots a month, no card needed. Paid plans start at $19 a month (https://curlshot.com/pricing).
- Never print the key, paste it into a file in the repository, or put it in a link you show the user. Send it in the `X-Access-Key` header (an `Authorization: Bearer <key>` header or an `access_key` query parameter also work).
- Only public pages work out of the box. Local and private network addresses (localhost, 192.168.x.x, ...) are refused: for a local page, send its HTML instead (see step 4).

## Instructions

1. Check that `CURLSHOT_API_KEY` is set (`test -n "$CURLSHOT_API_KEY"`) without echoing its value.
2. Take the screenshot with a GET request:

   ```bash
   curl -sS -G "https://curlshot.com/api/v1/screenshot" \
     -H "X-Access-Key: $CURLSHOT_API_KEY" \
     --data-urlencode "url=https://example.com" \
     -d format=png \
     -D headers.txt \
     --output screenshot.png
   ```

   `-G` with `--data-urlencode` keeps the page address correctly encoded.
3. Add options as one more `-d name=value` each (GET) or as fields in a POST JSON body:

   | Want | Option |
   | --- | --- |
   | Whole page, top to bottom | `full_page=true` |
   | Phone or tablet view | `viewport_device=<preset>` (list: https://curlshot.com/docs/devices.md) |
   | Window size | `viewport_width=1440`, `viewport_height=900` |
   | Retina sharpness | `device_scale_factor=2` |
   | Dark theme | `dark_mode=true` |
   | One element only | `selector=<css selector>` |
   | Output format | `format=png` / `jpeg` / `webp` / `pdf` / `mp4` / `webm` / `gif` |
   | Hide cookie banners and ads | `block_cookie_banners=true`, `block_ads=true` |
   | Wait for content | `wait_until=networkidle`, `delay=2`, `wait_for_selector=<css selector>` |
   | Reuse a stored copy for free | `cache=true` |

   A misspelled option is rejected with `invalid_options` and a hint, never ignored. Every option: https://curlshot.com/docs/options.md
4. To render your own HTML or Markdown (or a local page), POST JSON:

   ```bash
   curl -sS -X POST "https://curlshot.com/api/v1/screenshot" \
     -H "X-Access-Key: $CURLSHOT_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"html": "<h1>Hello</h1>", "format": "png", "viewport_width": 1200, "viewport_height": 630}' \
     --output card.png
   ```

   Use `"markdown": "..."` instead of `html` to render Markdown as a styled page, and `"format": "pdf"` for a PDF.
5. Check the result before you use it (see Output), then tell the user where the file is.

## Output

- Success: HTTP `200` with a `Content-Type` of `image/*`, `video/*` or `application/pdf`. The body is the file, saved by `--output`.
- Failure: HTTP `400` or higher with `Content-Type: application/json`. Do not keep that file as an image; read it as an error (see Error Handling).
- Useful response headers: `X-Quota-Remaining` and `X-Quota-Reset` (screenshots left this period), `X-RateLimit-Remaining`, `X-Concurrency-Remaining`, `X-Cache` (cache hit or miss), and `X-Reference-Id` (quote it when contacting support).
- Billing: one successful render counts as one screenshot, whatever its size. Failed renders and cache hits are free.

## Error Handling

An error body looks like this:

```json
{
  "error_code": "invalid_options",
  "error_message": "fullpage: is not a known option; did you mean `full_page`?",
  "documentation_url": "https://curlshot.com/docs/errors#invalid_options",
  "errors": [{ "field": "fullpage", "message": "is not a known option; did you mean `full_page`?" }]
}
```

Branch on `error_code`, never on `error_message`.

| Status | Codes | What to do |
| --- | --- | --- |
| `400`, `413`, `422` | `invalid_options`, `invalid_url`, `invalid_request`, `content_too_large`, `selector_not_found`, `page_blocked` | Do not retry. Fix the request using `errors` and the hint. |
| `401` | `access_key_required`, `access_key_invalid` | Do not retry. Ask the user to check `CURLSHOT_API_KEY`. |
| `402` | `quota_exceeded` | Do not retry. Tell the user the quota is used up and show `upgrade_url`. |
| `403` | `host_not_allowed`, `feature_not_available`, `email_not_verified` | Do not retry. For a private or local address, send the HTML instead. For a plan feature, show `upgrade_url`. |
| `429` | `rate_limited`, `concurrency_limit`, `queue_limit`, `failure_limit` | Wait the seconds in `Retry-After`, then retry once. |
| `500`, `502`, `503`, `504` | `internal_error`, `navigation_failed`, `timeout`, `renderer_busy`, `renderer_unavailable`, `service_unavailable` | Retry once after a short pause. For `timeout`, drop `wait_until=networkidle` (the default is `load`) or lower `delay`. |

Every code with its cause and fix: https://curlshot.com/docs/errors.md

## Examples

Full-page screenshot of a site on an iPhone, cookie banners hidden:

```bash
curl -sS -G "https://curlshot.com/api/v1/screenshot" \
  -H "X-Access-Key: $CURLSHOT_API_KEY" \
  --data-urlencode "url=https://example.com" \
  -d viewport_device=iphone_15_pro -d full_page=true -d block_cookie_banners=true \
  --output mobile.png
```

A4 PDF of a page:

```bash
curl -sS -G "https://curlshot.com/api/v1/screenshot" \
  -H "X-Access-Key: $CURLSHOT_API_KEY" \
  --data-urlencode "url=https://example.com/invoice" \
  -d format=pdf -d pdf_paper_format=a4 \
  --output page.pdf
```

Markdown rendered to a PDF:

```bash
curl -sS -X POST "https://curlshot.com/api/v1/screenshot" \
  -H "X-Access-Key: $CURLSHOT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"markdown": "# Release notes\n\n- Faster renders", "format": "pdf"}' \
  --output notes.pdf
```

Check the status before trusting the file:

```bash
status=$(curl -sS -G "https://curlshot.com/api/v1/screenshot" \
  -H "X-Access-Key: $CURLSHOT_API_KEY" \
  --data-urlencode "url=https://example.com" \
  -o shot.png -w '%{http_code}')
[ "$status" = 200 ] || { cat shot.png; rm shot.png; }
```

## Limitations

- Needs a CurlShot access key in `CURLSHOT_API_KEY`. The free plan gives 100 screenshots a month; after that, requests fail with `402 quota_exceeded` until the quota resets or the plan is upgraded.
- Only public addresses can be captured by URL. Localhost and private network addresses are refused; send the page's HTML instead.
- Every successful render calls an external paid service (curlshot.com) and counts against the user's quota.

## Resources

- Docs: https://curlshot.com/docs
- Getting started: https://curlshot.com/docs/getting-started.md
- All options: https://curlshot.com/docs/options.md
- Device presets: https://curlshot.com/docs/devices.md
- Errors: https://curlshot.com/docs/errors.md
- OpenAPI description: https://curlshot.com/openapi.json
- Everything in one file: https://curlshot.com/llms-full.txt
- Pricing: https://curlshot.com/pricing
- MCP alternative: agents with MCP support can use the hosted server `https://curlshot.com/api/mcp` (streamable HTTP, `Authorization: Bearer <key>`) with the tools `take_screenshot` and `list_devices`. Setup per client is in this repository's README.
