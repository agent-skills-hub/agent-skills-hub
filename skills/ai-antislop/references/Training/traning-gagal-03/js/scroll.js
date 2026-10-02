/* scroll.js — scroll engine: progress, smoothed follow value,
   velocity, reveals, per-section progress. Passive listeners only,
   layout reads batched in rAF. */

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lerp = (a, b, t) => a + (b - a) * t;

export function createScroll() {
  const doc = document.documentElement;
  const s = {
    y: window.scrollY,
    vh: window.innerHeight,
    max: 1,
    progress: 0,   // raw 0..1 page progress
    smooth: 0,     // critically-damped-ish follow value for 3D
    velocity: 0,   // smoothed px/frame scroll velocity (signed)
    reduced: matchMedia('(prefers-reduced-motion: reduce)').matches,
  };

  let lastY = s.y;
  let velSm = 0;

  function measure() {
    s.y = window.scrollY;
    s.vh = window.innerHeight;
    s.max = Math.max(1, doc.scrollHeight - s.vh);
    s.progress = clamp01(s.y / s.max);
  }

  // call every rAF with dt in seconds
  function update(dt) {
    measure();
    const rawVel = s.y - lastY;
    lastY = s.y;
    // smooth velocity (frame-rate independent-ish)
    const k = 1 - Math.pow(0.001, dt);
    velSm = lerp(velSm, rawVel, Math.min(1, k * 3));
    s.velocity = velSm;
    // smoothed progress follows raw with slight lag = buttery 3D
    s.smooth = lerp(s.smooth, s.progress, s.reduced ? 1 : 1 - Math.pow(0.0001, dt));
    if (Math.abs(s.smooth - s.progress) < 0.0004) s.smooth = s.progress;
  }

  // progress of an element's journey through the viewport: 0 = just
  // entering bottom, 1 = just leaving top
  function through(el) {
    const r = el.getBoundingClientRect();
    const total = s.vh + r.height;
    return clamp01((s.vh - r.top) / total);
  }

  // reveal-on-scroll for .reveal (skipped entirely under reduced motion
  // via CSS; observer still harmless)
  function observeReveals() {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
    return io;
  }

  measure();
  return { s, update, through, observeReveals, clamp01, lerp };
}
