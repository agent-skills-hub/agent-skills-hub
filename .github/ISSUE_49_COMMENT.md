Suggested comment for [Issue #49](https://github.com/sickn33/agent-skills-hub/issues/49). Paste this on the issue:

---

The 404 happens because the package wasn’t published to npm yet. We’ve addressed it in two ways:

1. **Publish to npm** – We’re set up to publish so `npx agent-skills-hub` will work after the first release. You can also trigger a manual publish via the “Publish to npm” workflow (Actions tab) if you have `NPM_TOKEN` configured.

2. **Fallback** – Until then (or if you hit a 404 for any reason), use:
   ```bash
   npx github:sickn33/agent-skills-hub
   ```
   The README, GETTING_STARTED, and FAQ now mention this fallback.

Thanks for reporting.
