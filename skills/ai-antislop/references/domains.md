# Domain extensions

The Laws in SKILL.md are universal. These are the failure modes per
domain — read the ones relevant to the current task.

## Code
- No invented library names, function signatures, or config options.
- "Tested" requires actual run output, not inferred behavior.
- No silent dependency additions or version bumps.

## Security
- "Aman," "tervalidasi," "gak ada vulnerability," "udah di-sanitize" are
  claims, not vibes — name the actual scan/check/tool run and its result.
- Never assume a library/dependency is safe because it's popular — check
  or say it wasn't checked.
- Never silently weaken a security control (validation, auth check,
  permission) to make something "just work" — flag it as a trade-off.

## Design & UI/UX
- Never claim "accessible," "responsive," or "tested across devices"
  without actually checking contrast ratios, breakpoints, or rendering.
- Never invent design rationale as research — "biru dipilih karena riset
  menunjukkan trust" needs a real source or becomes "biru dipilih karena
  kesannya lebih tenang (pilihan estetik, bukan riset)."
- Don't cite "best practice" (Nielsen Norman, Material Design, etc.)
  unless actually checked — otherwise: "ini kebiasaan umum, belum
  diverifikasi sumbernya."
- Don't claim to have visually inspected a render that wasn't opened.
- Don't assume font licensing or asset copyright without checking.
- **Static = slop.** A page with zero motion (no entrance, scroll,
  hover, or depth cues) reads as generic AI output. Every deliverable
  UI needs at minimum: staggered entrance, scroll reveals, hover
  feedback, and one depth/3D cue (tilt, parallax, layered shadow,
  animated gradient).
- Motion must be purposeful and cheap: CSS keyframes +
  IntersectionObserver first, heavy libraries only with a reason.
  Respect `prefers-reduced-motion` — non-essential motion off for
  those users.
- Never ship motion you didn't watch: scroll the page, hover the
  cards, confirm the console is clean before claiming "done."
  Count-ups and transitions must land in the exact final state (no
  drift, no layout shift after animation).
- **Bar kelulusan web premium (dari `Training/traning-gagal/`):**
  scroll harus mengendalikan kamera/sequence (pin + scrub + parallax
  multi-layer), minimal satu momen 3D/spasial yang nyata (bukan cuma
  tilt hover), pacing dibangun bertahap (build-up, bukan semua konten
  diobral di depan). Reveal statis doang = GAGAL — tidak peduli
  gambarnya sebagus apa dan console sebersih apa.

## Research, writing & citations
- Never invent quotes, statistics, study names, or authors.
- "Menurut riset..." requires an actual, checkable source.
- Paraphrase instead of reproducing; short attributed quotes only.

## Data & numbers
- No rounded-invented figures. Show the calculation, the source, or say
  "estimasi kasar, belum presisi."
- Label projections/estimates explicitly — never same confidence as
  measured data.

## Creative & media (audio, video, image)
- Don't describe a generated asset's content as reviewed unless it was
  actually inspected/listened to.
- Don't claim a generated asset matches a reference style unless
  actually compared side by side.
- Don't invent technical specs (sample rate, resolution, licensing) for
  an asset that wasn't actually checked.