# WebGL Tuning

Source: [naughtyduk/liquidGL](https://github.com/naughtyduk/liquidGL), MIT license.

## Parameter Guide

| Parameter | Start | Change carefully |
| --- | ---: | --- |
| `resolution` | `1.5` | Lower to `1` for mobile or several lenses; raise to `2` only after profiling. |
| `refraction` | `0.007` | Keep below `0.012` for readable product UI. |
| `aberration` | `0.03` | Keep subtle; high values create colored text-like ghosts. |
| `bevelDepth` | `0.075` | Controls edge depth and highlight strength. |
| `bevelWidth` | `0.18` | Increase slightly for large panels, reduce for compact cards. |
| `frost` | `2.2` | Increase for privacy/readability; reduce to reveal more background detail. |
| `magnify` | `1.006` | Avoid obvious zoom on operational surfaces. |

## Integration Rules

- Use one visual background as `snapshot`; exclude foreground content and overlays.
- Keep the target empty and the content in a sibling layer with higher `z-index`.
- Initialize after images and fonts needed by the snapshot have loaded.
- Call `liquidGL.registerDynamic(selector)` for animated or replaced backgrounds.
- Prefer a small number of lenses. Profile GPU use before rendering glass on long lists.
- Do not enable tilt by default. It adds motion and another stacking layer.

## Failure Signatures

- **Duplicate text:** Content was placed inside the WebGL target. Move it to the sibling overlay.
- **Detached or stale image:** The snapshot is wrong or dynamic content was not registered.
- **Black/blank lens:** WebGL or capture failed. Keep the CSS fallback visible and inspect console errors.
- **Hard color bands:** Reduce refraction/aberration or increase frost.
- **Lens clips while moving:** Check ancestor overflow and target stacking context.
