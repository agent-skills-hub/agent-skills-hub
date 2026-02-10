# Changelog

All notable changes to the **Antigravity Awesome Skills** collection are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.7.0] - 2026-02-03 - "Installer Fix & OpenCode Docs"

> Critical installer fix for Windows and OpenCode documentation completion.

### Fixed

- **Installer**: Resolved `ReferenceError` for `tagArg` variable in `bin/install.js` ensuring correct execution on Windows/PowerShell (PR #53).

### Documentation

- **OpenCode**: Completed documentation for OpenCode integration in `README.md`.

---

## [4.6.0] - 2026-02-01 - "SPDD & Radix UI Design System"

> Agent workflow docs (SPDD) and Radix UI design system skill.

### Added

- **New Skill**: `radix-ui-design-system` – Build accessible design systems with Radix UI primitives (headless, theming, WCAG, examples).
- **Docs**: `skills/SPDD/` – Research, spec, and implementation workflow docs (1-research.md, 2-spec.md, 3-implementation.md).

### Registry

- **Total Skills**: 626 (from 625). Catalog regenerated.

---

## [4.5.0] - 2026-01-31 - "Stitch UI Design"

> Expert prompting guide for Google Stitch AI-powered UI design tool.

### Added

- **New Skill**: `stitch-ui-design` – Expert guide for creating effective prompts for Google Stitch AI UI design tool (Gemini 2.5 Flash). Covers prompt structure, specificity techniques, iteration strategies, design-to-code workflows, and 10+ examples for landing pages, mobile apps, and dashboards.

### Changed

- **Documentation**: Clarified in README.md and GETTING_STARTED.md that installation means cloning the full repo once; Starter Packs are curated lists to discover skills by role, not a different installation method (fixes [#44](https://github.com/sickn33/agent-skills-hub/issues/44)).

### Registry

- **Total Skills**: 625 (from 624). Catalog regenerated.

### Credits

- [@ALEKGG1](https://github.com/ALEKGG1) – stitch-ui-design (PR #45)
- [@CypherPoet](https://github.com/CypherPoet) – Documentation clarity (#44)

---

## [4.4.0] - 2026-01-30 - "fp-ts skills for TypeScript"

> Three practical fp-ts skills for TypeScript functional programming.

### Added

- **New Skills** (from [whatiskadudoing/fp-ts-skills](https://github.com/whatiskadudoing/fp-ts-skills)):
  - `fp-ts-pragmatic` – Pipe, Option, Either, TaskEither without academic jargon.
  - `fp-ts-react` – Patterns for fp-ts with React 18/19 and Next.js 14/15 (state, forms, data fetching).
  - `fp-ts-errors` – Type-safe error handling with Either and TaskEither.

### Registry

- **Total Skills**: 624 (from 621). Catalog regenerated.

---

## [4.3.0] - 2026-01-29 - "VoltAgent Integration & Context Engineering Suite"

> 61 new skills from VoltAgent/awesome-agent-skills: official team skills and context engineering suite.

### Added

- **61 new skills** from [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills):
  - **Official (27)**: Sentry (commit, create-pr, find-bugs, iterate-pr), Trail of Bits (culture-index, fix-review, sharp-edges), Expo (expo-deployment, upgrading-expo), Hugging Face (hugging-face-cli, hugging-face-jobs), Vercel, Google Stitch (design-md), Neon (using-neon), n8n (n8n-code-python, n8n-mcp-tools-expert, n8n-node-configuration), SwiftUI, fal.ai (fal-audio, fal-generate, fal-image-edit, fal-platform, fal-upscale, fal-workflow), deep-research, imagen, readme.
  - **Community (34)**: Context suite (context-fundamentals, context-degradation, context-compression, context-optimization, multi-agent-patterns, memory-systems, evaluation), frontend-slides, linear-claude-skill, skill-rails-upgrade, terraform-skill, tool-design, screenshots, automate-whatsapp, observe-whatsapp, aws-skills, ui-skills, vexor, pypict-skill, makepad-skills, threejs-skills, claude-scientific-skills, claude-win11-speckit-update-skill, security-bluebook-builder, claude-ally-health, clarity-gate, beautiful-prose, claude-speed-reader, skill-seekers, varlock-claude-skill, superpowers-lab, nanobanana-ppt-skills, x-article-publisher-skill, ffuf-claude-skill.

### Registry

- **Total Skills**: 614 (from 553). Catalog and SOURCES.md updated.

### Credits

- VoltAgent/awesome-agent-skills and official teams (Sentry, Trail of Bits, Expo, Hugging Face, Vercel Labs, Google Labs, Neon, fal.ai).

---

## [4.0.0] - 2026-01-28 - "The Enterprise Era"

> **A massive integration of 300+ Enterprise skills, transforming Antigravity into a complete operating system for AI agents.**

### Added

- **Massive Skill Injection**: Merged 300+ Enterprise skills from `rmyndharis/antigravity-skills`.
- **New Categories**:
  - **Architecture & Design**: `backend-architect`, `c4-architecture`.
  - **Data & AI**: `rag-engineer`, `langchain-architecture`.
  - **Security**: `security-auditor`, `cloud-pentesting`.
- **Catalog System**: Introduced `CATALOG.md` and `scripts/build-catalog.js` for automated, table-based skill discovery.

### Changed

- **Documentation Overhaul**:
  - Removed the legacy 250+ row skill table from `README.md`.
  - Restructured `README.md` to focus on high-level domains.
  - Replaced static registry with dynamic `CATALOG.md`.
- **Version Bump**: Major version update to 4.0.0 reflecting the doubling of skill capacity (247 -> 550+).

### Credits

- **[@rmyndharis](https://github.com/rmyndharis)** - For the massive contribution of 300+ Enterprise skills and valid catalog logic.
- **[@sstklen](https://github.com/sstklen)** & **[@rookie-ricardo](https://github.com/rookie-ricard)** - Continued community support.

## [3.4.0] - 2026-01-27 - "Voice Intelligence & Categorization"

### Added

- **New Skill**: `voice-ai-engine-development` - Complete toolkit for building real-time voice agents (OpenAI Realtime, Vapi, Deepgram, ElevenLabs).
- **Categorization**: Major README update introducing a concise "Features & Categories" summary table.

### Changed

- **README**: Replaced text-heavy category lists with a high-level summary table.
- **Registry**: Synced generic skill count (256) across documentation.

### Contributors

- [@sickn33](https://github.com/sickn33) - Voice AI Engine (PR #33)
- [@community](https://github.com/community) - Categorization Initiative (PR #32)

## [3.3.0] - 2026-01-26 - "News & Research"

### Added

- **New Skills**:
  - `last30days`: Research any topic from the last 30 days on Reddit + X + Web.
  - `daily-news-report`: Generate daily news reports from multiple sources.

### Changed

- **Registry**: Updated `skills_index.json` and `README.md` registry (Total: 255 skills).

## [3.2.0] - 2026-01-26 - "Clarity & Consistency"

### Changed

- **Skills Refactoring**: Significant overhaul of `backend-dev-guidelines`, `frontend-design`, `frontend-dev-guidelines`, and `mobile-design`.
  - **Consolidation**: Merged fragmented documentation into single, authoritative `SKILL.md` files.
  - **Final Laws**: Introduced "Final Laws" sections to provide strict, non-negotiable decision frameworks.
  - **Simplification**: Removed external file dependencies to improve context retrieval for AI agents.

### Fixed

- **Validation**: Fixed critical YAML frontmatter formatting issues in `seo-fundamentals`, `programmatic-seo`, and `schema-markup` that were blocking strict validation.
- **Merge Conflicts**: Resolved text artifact conflicts in SEO skills.

## [3.1.0] - 2026-01-26 - "Stable & Deterministic"

### Fixed

- **CI/CD Drift**: Resolved persistent "Uncommitted Changes" errors in CI by making the index generation script deterministic (sorting by name + ID).
- **Registry Sync**: Synced `README.md` and `skills_index.json` to accurately reflect all 253 skills.

### Added (Registry Restore)

The following skills are now correctly indexed and visible in the registry:

- **Marketing & Growth**: `programmatic-seo`, `schema-markup`, `seo-fundamentals`, `form-cro`, `popup-cro`, `analytics-tracking`.
- **Security**: `windows-privilege-escalation`, `wireshark-analysis`, `wordpress-penetration-testing`, `writing-plans`.
- **Development**: `tdd-workflow`, `web-performance-optimization`, `webapp-testing`, `workflow-automation`, `zapier-make-patterns`.
- **Maker Tools**: `telegram-bot-builder`, `telegram-mini-app`, `viral-generator-builder`.

### Changed

- **Documentation**: Added `docs/CI_DRIFT_FIX.md` as a canonical reference for resolving drift issues.
- **Guidance**: Updated `docs/GETTING_STARTED.md` counts to match the full registry (253+ skills).
- **Maintenance**: Updated `MAINTENANCE.md` with strict protocols for handling generated files.

## [3.0.0] - 2026-01-25 - "The Governance Update"

### Added

- **Governance & Security**:
  - `docs/QUALITY_BAR.md`: Defined 5-point validation standard (Metadata, Risk, Triggers).
  - `docs/SECURITY_GUARDRAILS.md`: Enforced "Authorized Use Only" for offensive skills.
  - `CODE_OF_CONDUCT.md`: Adhered to Contributor Covenant v2.1.
- **Automation**:
  - `scripts/validate_skills.py`: Automated Quality Bar enforcement (Soft Mode supported).
  - `.github/workflows/ci.yml`: Automated PR checks.
  - `scripts/generate_index.py`: Registry generation with Risk & Source columns.
- **Experience**:
  - `docs/BUNDLES.md`: 9 Starter Packs (Essentials, Security, Web, Agent, Game Dev, DevOps, Data, Testing, Creative).
  - **Interactive Registry**: README now features Risk Levels (🔴/🟢/🟣) and Collections.
- **Documentation**:
  - `docs/EXAMPLES.md`: Cookbook with 3 real-world scenarios.
  - `docs/SOURCES.md`: Legal ledger for attributions and licenses.
  - Release announcements are documented in this CHANGELOG.

### Changed

- **Standardization**: All 250+ skills are now validated against the new Quality Bar schema.
- **Project Structure**: Introduced `docs/` folder for scalable documentation.

## [2.14.0] - 2026-01-25 - "Web Intelligence & Windows"

### Added

- **New Skill**:
  - `context7-auto-research`: Auto-research capability for Claude Code.
  - `codex-review`: Professional code review with AI integration.
  - `exa-search`: Semantic search and discovery using Exa API.
  - `firecrawl-scraper`: Deep web scraping and PDF parsing.
  - `tavily-web`: Content extraction and research using Tavily.
  - `busybox-on-windows`: UNIX tool suite for Windows environments.

### Changed

- **Documentation**: Updated `obsidian-clipper-template-creator` docs and templates.
- **Index & Registry**: Updated `skills_index.json` and `README.md` registry.

### Fixed

- **Skills**: Fixed YAML frontmatter quoting in `lint-and-validate`.

## [2.13.0] - 2026-01-24 - "NoSQL Expert"

### Added

- **New Skill**:
  - `nosql-expert`: Expert guidance for distributed NoSQL databases (Cassandra, DynamoDB), focusing on query-first modeling and anti-patterns.

### Changed

- **Index & Registry**: Updated `skills_index.json` and `README.md` registry.

### Contributors

- [@sickn33](https://github.com/sickn33) - PR #23

## [2.12.0] - 2026-01-23 - "Enterprise & UI Power"

### Added

- **New Skills**:
  - `production-code-audit`: Comprehensive enterprise auditing skill for production readiness.
  - `avalonia-layout-zafiro`: Zafiro layout guidelines for Avalonia UI.
  - `avalonia-viewmodels-zafiro`: ViewModel composition patterns for Avalonia.
  - `avalonia-zafiro-development`: Core development rules for Avalonia Zafiro applications.

### Changed

- **Index & Registry**: Updated `skills_index.json` and `README.md` registry (Total: 243 skills).

### Contributors

- [@SuperJMN](https://github.com/SuperJMN) - PR #20
- [@Mohammad-Faiz-Cloud-Engineer](https://github.com/Mohammad-Faiz-Cloud-Engineer) - PR #21

## [2.11.0] - 2026-01-23 - "Postgres Performance"

### Added

- **New Skill**:
  - `postgres-best-practices`: Comprehensive Supabase PostgreSQL performance optimization guide with 30+ rules covering query performance, connection management, RLS security, schema design, locking, and monitoring.

### Changed

- **Official Sources**: Added [supabase/agent-skills](https://github.com/supabase/agent-skills) to Credits & Sources.
- **Index & Registry**: Updated `skills_index.json` and `README.md` registry (Total: 239 skills).

### Contributors

- [@ar27111994](https://github.com/ar27111994) - PR #19

---

## [2.10.0] - 2026-01-22 - "Developer Excellence"

### Added

- **New Skills**:
  - `api-security-best-practices`: Comprehensive guide for secure API design and defense.
  - `environment-setup-guide`: Systematic approach to project onboarding and tool configuration.
  - `web-performance-optimization`: Methodologies for optimizing Core Web Vitals and loading speed.

### Changed

- **Enhanced Skill**:
  - `code-review-checklist`: Replaced with a much more detailed and systematic version covering functionality, security, and quality.

### Fixed

- **Index & Registry**: Updated `skills_index.json` and `README.md` registry (Total: 238 skills).

### Added

- **Automation Support**:
  - `scripts/update_readme.py`: Automated script to sync skill counts and regenerate the registry table.
  - Updated `MAINTENANCE.md` to reflect the new automated workflow.
- **Repository Quality**:
  - `MAINTENANCE.md` is now tracked in the repository (removed from `.gitignore`).
  - Improved contribution guidelines.

## [2.8.0] - 2026-01-22 - "Documentation Power"

### Added

- **API Documentation Generator**: New skill to automatically generate comprehensive API documentation (`skills/api-documentation-generator`).
- **Remotion Best Practices**: 28 modular rules for programmatic video creation (`skills/remotion-best-practices`).

## [2.7.0] - 2026-01-22 - "Agent Memory"

### Added

- **Agent Memory MCP**: New skill providing persistent, searchable knowledge management for AI agents (`skills/agent-memory-mcp`).

### Changed

- **Renamed Skill**: `agent-memory` was renamed to `agent-memory-mcp` to avoid naming conflicts.

---

## [2.6.0] - 2026-01-21 - "Everything Skills Edition"

### Added

- **8 Verified Skills** from [affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code):
  - `cc-skill-backend-patterns`
  - `cc-skill-clickhouse-io`
  - `cc-skill-coding-standards`
  - `cc-skill-continuous-learning`
  - `cc-skill-frontend-patterns`
  - `cc-skill-project-guidelines-example`
  - `cc-skill-security-review`
  - `cc-skill-strategic-compact`
- **Documentation**: New `docs/WALKTHROUGH.md` for import process details.

### Changed

- **Skill Cleanup**: Removed 27 unwanted agents, commands, and rules from the `everything-claude-code` import to focus strictly on skills.
- **Index**: Regenerated `skills_index.json` (Total: 233 skills).
- **Credits**: Updated README credits and registry.

## [1.0.0] - 2026-01-19 - "Marketing Edition"

### Added

- **23 Marketing & Growth skills** from [coreyhaines31/marketingskills](https://github.com/coreyhaines31/marketingskills):
  - **CRO**: `page-cro`, `signup-flow-cro`, `onboarding-cro`, `form-cro`, `popup-cro`, `paywall-upgrade-cro`
  - **Content**: `copywriting`, `copy-editing`, `email-sequence`
  - **SEO**: `seo-audit`, `programmatic-seo`, `schema-markup`, `competitor-alternatives`
  - **Paid**: `paid-ads`, `social-content`
  - **Growth**: `referral-program`, `launch-strategy`, `free-tool-strategy`
  - **Analytics**: `ab-test-setup`, `analytics-tracking`
  - **Strategy**: `pricing-strategy`, `marketing-ideas`, `marketing-psychology`
- New "Marketing & Growth" category in Features table

### Changed

- Total skills count: **179**

---

## [0.7.0] - 2026-01-19 - "Education Edition"

### Added

- **Moodle External API Development** skill via PR #6
- Comprehensive Moodle LMS web service API development

### Changed

- Total skills count: **156**

---

## [0.6.0] - 2026-01-19 - "Vibeship Integration"

### Added

- **57 skills** from [vibeforge1111/vibeship-spawner-skills](https://github.com/vibeforge1111/vibeship-spawner-skills):
  - AI Agents category (~30 skills)
  - Integrations & APIs (~25 skills)
  - Maker Tools (~11 skills)
- Alphabetically sorted skill registry

### Changed

- Total skills count: **155**

---

## [0.5.0] - 2026-01-18 - "Agent Manager"

### Added

- **Agent Manager Skill** - Multi-agent orchestration via tmux
- Major repository expansion with community contributions

### Changed

- Total skills count: **131**

---

## [0.4.0] - 2026-01-18 - "Security Fortress"

### Added

- **60+ Cybersecurity skills** from [zebbern/claude-code-guide](https://github.com/zebbern/claude-code-guide):
  - Ethical Hacking Methodology
  - Metasploit Framework
  - Burp Suite Testing
  - SQLMap, Active Directory, AWS Pentesting
  - OWASP Top 100 Vulnerabilities
  - Red Team Tools
- `claude-code-guide` skill

### Changed

- Total skills count: ~90

---

## [0.3.0] - 2026-01-17 - "First Stable Registry"

### Added

- Complete skill registry table in README
- GitHub workflow automation
- SEO optimizations

### Changed

- Total skills count: **71**

---

## [0.2.0] - 2026-01-16 - "Official Skills"

### Added

- **Official Anthropic skills** integration
- **Vercel Labs skills** integration
- BlockRun: Agent Wallet for LLM Micropayments
- 7 new skills from GitHub analysis

### Changed

- Total skills count: **~65**

---

## [0.1.0] - 2026-01-15 - "Initial Release"

### Added

- **58 core skills** aggregated from community:
  - [obra/superpowers](https://github.com/obra/superpowers) - Original Superpowers
  - [guanyang/antigravity-skills](https://github.com/guanyang/antigravity-skills) - Core extensions
  - [diet103/claude-code-infrastructure-showcase](https://github.com/diet103/claude-code-infrastructure-showcase) - Infrastructure skills
  - [ChrisWiles/claude-code-showcase](https://github.com/ChrisWiles/claude-code-showcase) - React UI patterns
  - [travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills) - Loki Mode
  - [alirezarezvani/claude-skills](https://github.com/alirezarezvani/claude-skills) - Senior Engineering
- Universal **SKILL.md** format
- Compatibility with Claude Code, Gemini CLI, Cursor, Copilot, Antigravity

---

## Credits

See [README.md](README.md#credits--sources) for full attribution.

## License

MIT License - See [LICENSE](LICENSE) for details.
