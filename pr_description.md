# Pull Request Description

Hey @legendaryabhi 👋

I ran your skills through `tessl skill review` at work and found some targeted improvements. Here's the full before/after:

| Skill | Before | After | Change |
|-------|--------|-------|--------|
| api-design-principles | 61% | 89% | +28% |
| python-pro | 39% | 70% | +31% |
| react-patterns | 52% | 76% | +24% |
| kubernetes-architect | 48% | 70% | +22% |
| docker-expert | 72% | 76% | +4% |

![Score Card](score_card.png)

<details>
<summary>What changed</summary>

**api-design-principles** — Rewrote description with concrete actions and trigger terms (endpoints, OpenAPI, Swagger). Added inline REST naming conventions table, error response schema, and cursor-based pagination example. Linked existing reference files for progressive disclosure.

**python-pro** — Replaced vague "Master/Expert" description with specific actions and trigger terms (.py, uv, ruff, pytest). Removed persona statement. Added Quick Start section with executable `uv init` + `pyproject.toml` config and a validation workflow with real commands.

**react-patterns** — Added "Use when..." triggers with natural terms (useState, useEffect, .tsx, Zustand). Added concrete compound component code example and useDebounce custom hook extraction example.

**kubernetes-architect** — Replaced resume-style description with action verbs and specific triggers. Removed persona statement. Added validation workflow with `kubectl diff`, rollout status, and rollback commands.

**docker-expert** — Tightened description to use action verbs. Removed persona paragraph, redundant code review checklist, diagnostics section, and integration guidelines. Replaced with a concise 5-point validation checklist.

</details>

I kept this PR focused on 5 skills with the biggest improvements to keep the diff reviewable. Happy to follow up with more in a separate PR if you'd like.

Honest disclosure — I work at @tesslio where we build tooling around skills like these. Not a pitch - just saw room for improvement and wanted to contribute.

Want to self-improve your skills? Just point your agent (Claude Code, Codex, etc.) at [this Tessl guide](https://docs.tessl.io/evaluate/optimize-a-skill-using-best-practices) and ask it to optimize your skill. Ping me - [@yogesh-tessl](https://github.com/yogesh-tessl) - if you hit any snags.

Thanks in advance 🙏

## Quality Bar Checklist

**All items must be checked before merging.**

- [x] **Standards**: I have read `docs/QUALITY_BAR.md` and `docs/SECURITY_GUARDRAILS.md`.
- [x] **Metadata**: The `SKILL.md` frontmatter is valid (checked with `scripts/validate_skills.py`).
- [x] **Risk Label**: I have assigned the correct `risk:` tag (`none`, `safe`, `critical`, `offensive`).
- [x] **Triggers**: The "When to use" section is clear and specific.
- [ ] **Security**: If this is an _offensive_ skill, I included the "Authorized Use Only" disclaimer.
- [x] **Local Test**: I have verified the skill works locally.
- [ ] **Credits**: I have added the source credit in `README.md` (if applicable).

## Type of Change

- [ ] New Skill (Feature)
- [x] Documentation Update
- [ ] Infrastructure
