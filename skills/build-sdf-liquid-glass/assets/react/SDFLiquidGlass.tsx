import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { Vaso } from 'vaso'

export interface SDFLiquidGlassProps {
  children: ReactNode
  className?: string
  height?: number
  radius?: number
  depth?: number
  blur?: number
  dispersion?: number
  style?: CSSProperties
}

export function SDFLiquidGlass({
  children,
  className = '',
  height = 86,
  radius = 16,
  depth = 0.58,
  blur = 0.4,
  dispersion = 0.32,
  style
}: SDFLiquidGlassProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(1)

  useLayoutEffect(() => {
    const host = hostRef.current
    if (!host) return

    const update = () => setWidth(Math.max(1, Math.round(host.getBoundingClientRect().width)))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(host)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={hostRef}
      className={`sdf-liquid-glass ${className}`.trim()}
      style={{ ...style, minHeight: height, borderRadius: radius }}
    >
      <Vaso
        className="sdf-liquid-glass__effect"
        width={width}
        height={height}
        radius={radius}
        depth={depth}
        blur={blur}
        dispersion={dispersion}
      >
        <div style={{ width, height }} />
      </Vaso>
      <div className="sdf-liquid-glass__content">{children}</div>
    </div>
  )
}
