/* main.js — Bukit Wisteria wiring: progress, hero type, reveals,
   word scrub, play-band orbit drive, lantern click (raycast),
   drag-orbit inspection, cursor light, loop, resize, fallbacks. */
import { initScene } from './scene.js';
import { createScroll } from './scroll.js';

document.documentElement.classList.add('js');

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobile = matchMedia('(max-width: 760px)').matches;
const S = createScroll();
const { s } = S;

const bar = document.getElementById('progress');

/* hero per-character entrance */
const hn = document.getElementById('heroName');
if (hn && !reduced) {
  hn.innerHTML = [...hn.textContent].map((c, i) =>
    `<span class="ch" style="display:inline-block;opacity:0;` +
    `animation:charin .8s cubic-bezier(.2,.8,.2,1) forwards;` +
    `animation-delay:${0.3 + i * 0.1}s">${c}</span>`).join('');
}
const st = document.createElement('style');
st.textContent = '@keyframes charin{from{opacity:0;transform:translateY(70px) rotate(5deg)}to{opacity:1;transform:none}}';
document.head.appendChild(st);

S.observeReveals();

/* word scrub */
const wp = document.getElementById('wordPara');
let words = [];
if (wp) {
  wp.innerHTML = wp.textContent.trim().split(/\s+/).map((w) => `<span class="w">${w}</span>`).join(' ');
  words = [...wp.querySelectorAll('.w')];
}
function lightWords() {
  if (!words.length || reduced) { words.forEach((w) => w.classList.add('lit')); return; }
  const r = wp.getBoundingClientRect();
  const p = Math.min(1, Math.max(0, (s.vh * 0.85 - r.top) / (s.vh * 0.55)));
  const n = Math.floor(p * words.length);
  words.forEach((w, i) => w.classList.toggle('lit', i < n));
}

/* play band: scroll drives extra orbit + meter */
const playBand = document.getElementById('playBand');
const playMeter = document.getElementById('playMeter');
let bandT = 0;
function driveBand() {
  if (!playBand) return;
  bandT = S.through(playBand);
  if (playMeter) playMeter.style.width = (bandT * 100).toFixed(1) + '%';
}

/* 3D scene */
let scene3d = null;
try {
  scene3d = initScene(document.getElementById('gl'), { mobile });
  const info = scene3d.stats();
  console.info(`[wisteria] WebGL diorama live — petals:${info.petals} calls:${info.calls}`);
} catch (err) {
  document.body.classList.add('no-gl');
  console.warn('[wisteria] WebGL unavailable, static fallback.', err);
}

/* cursor-reactive light */
addEventListener('pointermove', (e) => {
  if (!scene3d || reduced) return;
  scene3d.setPointer((e.clientX / window.innerWidth) * 2 - 1,
    -((e.clientY / window.innerHeight) * 2 - 1));
}, { passive: true });

/* drag = orbit inspection (stays where left, damped) */
let dragging = false, lx = 0, ly = 0, moved = 0;
addEventListener('pointerdown', (e) => {
  if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
  if (e.target.closest('a,button,input,nav')) return;
  dragging = true; moved = 0; lx = e.clientX; ly = e.clientY;
});
addEventListener('pointermove', (e) => {
  if (!dragging || !scene3d || reduced) return;
  const dx = e.clientX - lx;
  moved += Math.abs(dx);
  scene3d.addOrbit(dx);
  lx = e.clientX; ly = e.clientY;
}, { passive: true });
/* lantern toggle: shared by 3D-click and the DOM button */
const lanternBtn = document.getElementById('lanternBtn');
function setLanternUI(lit) {
  document.getElementById('lanternState').textContent =
    lit ? '🏮 Lentera: menyala (senja)' : '🌅 Lentera: padam (fajar)';
  if (lanternBtn) lanternBtn.textContent = lit ? 'Padamkan lentera' : 'Nyalakan lentera';
}
if (lanternBtn) lanternBtn.addEventListener('click', () => {
  if (scene3d && !reduced) setLanternUI(scene3d.toggleDusk());
});
addEventListener('pointerup', (e) => {
  // treat as lantern click only if it was a tap, not a drag
  if (dragging && moved < 6 && scene3d && !reduced) {
    const nx = (e.clientX / window.innerWidth) * 2 - 1;
    const ny = -((e.clientY / window.innerHeight) * 2 - 1);
    if (scene3d.clickLantern(nx, ny)) setLanternUI(scene3d.toggleDusk());
  }
  dragging = false;
});

/* magnetic buttons */
if (!reduced && matchMedia('(pointer:fine)').matches) {
  document.querySelectorAll('.btn').forEach((b) => {
    b.addEventListener('mousemove', (e) => {
      const r = b.getBoundingClientRect();
      b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * 0.18}px,${(e.clientY - r.top - r.height / 2) * 0.28}px)`;
    });
    b.addEventListener('mouseleave', () => { b.style.transform = ''; });
  });
}

/* main loop */
let last = performance.now();
let firstFrames = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  S.update(dt);
  bar.style.width = (s.progress * 100) + '%';
  lightWords();
  driveBand();
  if (scene3d) {
    if (reduced) {
      if (firstFrames === 0) scene3d.update({ smooth: s.smooth, vel: 0, spin: bandT }, 0, 0, true);
      firstFrames++;
    } else {
      scene3d.update({ smooth: s.smooth, vel: s.velocity, spin: bandT }, dt, now / 1000, false);
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

addEventListener('resize', () => { if (scene3d) scene3d.resize(); });
