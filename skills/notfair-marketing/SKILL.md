---
name: notfair-marketing
description: "SEO, GEO, Google Ads, and Meta Ads agent skills that connect to live account data via Google Ads MCP, Meta Ads MCP, Google Search Console MCP, and Google Analytics (GA4) MCP."
risk: safe
source: https://github.com/nowork-studio/NotFair
---

# NotFair Marketing Skills

NotFair is an open-source set of Claude Code skills for performance marketing and SEO. It connects to live account data through four MCP integrations: Google Ads MCP, Meta Ads MCP, Google Search Console MCP, and Google Analytics (GA4) MCP.

Skill areas:

- **seo/** — site analysis, keyword research, meta tags, schema markup, GEO (generative engine optimization), and content writing
- **google-ads/** — account audits, wasted-spend detection, search-term cleanup, keyword and bid management
- **meta-ads/** — Meta (Facebook + Instagram) Ads: ROAS analysis, creative fatigue detection, audience overlap

## When to Use

Use this skill when the user wants to:

- Audit or optimize a Google Ads or Meta Ads account with live data
- Detect wasted spend, poor-ROAS ad sets, or creative fatigue
- Run keyword research, meta-tag optimization, schema markup, or GEO optimization
- Pull live analytics from Google Search Console or Google Analytics (GA4)
- Manage bids, keywords, or campaign structure in Google Ads
- Analyze Meta audience overlap or ad creative performance

## Setup

1. Install from the NotFair repo:

   ```bash
   git clone https://github.com/nowork-studio/NotFair.git .claude/skills/notfair
   ```

2. Configure your MCP connections in `.claude/settings.json` — the repo README covers Google Ads MCP, Meta Ads MCP, Google Search Console MCP, and Google Analytics (GA4) MCP setup.

3. In Claude Code, invoke a skill directly:

   ```
   /notfair:google-ads-audit
   /notfair:seo-audit
   /notfair:meta-ads-audit
   ```

## Examples

### Audit Google Ads for wasted spend

```
/notfair:google-ads
> Find keywords with spend > $50 and zero conversions in the last 30 days
```

The skill connects to Google Ads MCP, runs a GAQL query, and returns a ranked list of underperformers with recommended actions.

### Check Meta Ads creative fatigue

```
/notfair:meta-ads
> Which ads have frequency > 4 and declining CTR over the last 14 days?
```

The skill connects to Meta Ads MCP via the Graph API, pulls ad-set metrics, and surfaces fatigued creatives with replacement suggestions.

### SEO + GEO keyword research

```
/notfair:keyword-research
> Find GEO-optimized keyword clusters for [your topic]
```

Pulls live search data via Google Search Console MCP and Google Analytics (GA4) MCP to surface queries with citation potential.

## Limitations

- Requires active MCP server connections for Google Ads MCP, Meta Ads MCP, Google Search Console MCP, and Google Analytics (GA4) MCP — skills are read-only without credentials.
- Google Ads and Meta Ads write operations (pausing keywords, updating budgets) require appropriate API permissions on the connected account.
- Does not support LinkedIn Ads, TikTok Ads, or Twitter/X Ads at this time.
- GEO optimization suggestions are advisory; actual AI citation depends on the target platform's indexing behavior.
