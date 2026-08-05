# SDF Tuning

Source: [huozhi/vaso](https://github.com/huozhi/vaso), MIT license. Vaso is based on Shuding's liquid-glass experiment.

## Parameter Guide

| Parameter | Start | Change carefully |
| --- | ---: | --- |
| `depth` | `0.58` | Use `0.35-0.8` for readable UI. Values near `2` are decorative. |
| `blur` | `0.4` | Increase for privacy, but keep text outside the filtered layer. |
| `dispersion` | `0.32` | Reduce first when colored edge fringes become distracting. |
| `radius` | `16` | Match the host radius exactly to avoid exposed corners. |

## Integration Rules

- Keep Vaso in an absolute background layer.
- Render text, icons, buttons, and focus rings in an unfiltered sibling overlay.
- Measure responsive width with `ResizeObserver`; pass explicit width and height to Vaso.
- Keep a CSS tint and backdrop blur under the effect as a browser fallback.
- Avoid remounting the effect on every pointer move; update position with the host element.

## Failure Signatures

- **Text looks doubled or rainbow-split:** Foreground content is inside the filtered layer.
- **Map is cropped or stretched:** Vaso dimensions do not match the host after resize.
- **Corners leak:** Radius differs between host, effect, and `[data-vaso]` layer.
- **Card is too milky:** Reduce CSS tint or Vaso blur before increasing depth.
