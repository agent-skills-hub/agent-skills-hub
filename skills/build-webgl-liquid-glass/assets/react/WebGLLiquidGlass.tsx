import { useEffect, useId, type CSSProperties, type ReactNode } from 'react'
import liquidGL from 'liquid-gl'

const initializedTargets = new Set<string>()

export interface WebGLLiquidGlassProps {
  children: ReactNode
  className?: string
  snapshotSelector?: string
  dynamicSelector?: string
  style?: CSSProperties
}

export function WebGLLiquidGlass({
  children,
  className = '',
  snapshotSelector = 'body',
  dynamicSelector,
  style
}: WebGLLiquidGlassProps) {
  const targetId = `liquid-gl-${useId().replace(/:/g, '')}`

  useEffect(() => {
    if (initializedTargets.has(targetId)) return

    const timer = window.setTimeout(() => {
      liquidGL({
        target: `#${targetId}`,
        snapshot: snapshotSelector,
        resolution: 1.5,
        refraction: 0.007,
        aberration: 0.03,
        bevelDepth: 0.075,
        bevelWidth: 0.18,
        frost: 2.2,
        shadow: false,
        specular: true,
        reveal: 'none',
        tilt: false,
        magnify: 1.006
      })
      initializedTargets.add(targetId)
      if (dynamicSelector) liquidGL.registerDynamic(dynamicSelector)
    }, 80)

    return () => window.clearTimeout(timer)
  }, [dynamicSelector, snapshotSelector, targetId])

  return (
    <div className={`webgl-liquid-glass ${className}`.trim()} style={style}>
      <div id={targetId} className="webgl-liquid-glass__lens" aria-hidden="true" />
      <div className="webgl-liquid-glass__content">{children}</div>
    </div>
  )
}
