/* main.js — No.2 / Umi Asanagi build. UI + interaction wiring:
   progress, hero type, marquee, reveals, word scrub, transformation
   scale, chapter lighting, LN⇄Anime slider, drag-to-inspect,
   cursor lights, loop, resize, fallbacks. */
import { initScene } from './scene.js';
import { createScroll } from './scroll.js';

document.documentElement.classList.add('js');

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobile = matchMedia('(max-width: 760px)').matches;
const S = createScroll();
const { s } = S;

/* ---------- progress bar ---------- */
const bar = document.getElementById('progress');

/* ---------- hero per-character entrance ---------- */
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

/* ---------- marquee seamless loop ---------- */
const track = document.getElementById('mtrack');
if (track) track.innerHTML += track.innerHTML;

/* ---------- reveals ---------- */
S.observeReveals();

/* ---------- word scrub ---------- */
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

/* ---------- transformation stage: scroll drives scale ---------- */
const transStage = document.getElementById('transStage');
const transImg = document.getElementById('transImg');
function scaleTransStage() {
  if (!transStage || reduced) return;
  const p = S.through(transStage);
  const sc = 0.82 + Math.sin(Math.min(1, p * 1.15) * Math.PI) * 0.26;
  transImg.style.transform = `scale(${sc.toFixed(4)})`;
}

/* ---------- 3D chapters: light the active card ---------- */
const chapters = [...document.querySelectorAll('#artefak .chapter')];
const cards = chapters.map((c) => c.querySelector('.card'));
let chapterT = 0;
function lightChapters() {
  const sec = document.getElementById('artefak');
  chapterT = S.through(sec);
  const idx = Math.min(chapters.length - 1, Math.floor(chapterT * chapters.length));
  cards.forEach((c, i) => {
    c.classList.toggle('lit', i === idx);
    c.classList.toggle('dim', i !== idx);
  });
}

/* ---------- LN ⇄ Anime compare slider (original interaction) ---------- */
const cmp = document.getElementById('compare');
const cmpTop = document.getElementById('cmpTop');
const cmpHandle = document.getElementById('cmpHandle');
const cmpRange = document.getElementById('cmpRange');
function setCmp(v) {
  v = Math.min(100, Math.max(0, Number(v)));
  cmpTop.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  cmpHandle.style.left = v + '%';
  if (document.activeElement !== cmpRange) cmpRange.value = v;
}
if (cmp) {
  setCmp(50);
  cmpRange.addEventListener('input', (e) => setCmp(e.target.value));
  let cdown = false;
  const slideTo = (clientX) => {
    const r = cmp.getBoundingClientRect();
    setCmp(((clientX - r.left) / r.width) * 100);
  };
  cmp.addEventListener('pointerdown', (e) => { cdown = true; slideTo(e.clientX); });
  addEventListener('pointermove', (e) => { if (cdown) slideTo(e.clientX); }, { passive: true });
  addEventListener('pointerup', () => { cdown = false; });
}

/* ---------- 3D scene (real WebGL, vendored three.js) ---------- */
let scene3d = null;
try {
  scene3d = initScene(document.getElementById('gl'), { mobile });
  const info = scene3d.stats();
  console.info(`[no2] WebGL scene live — petals:${info.petals} calls:${info.calls} tris:${info.tris}`);
} catch (err) {
  document.body.classList.add('no-gl');
  console.warn('[no2] WebGL unavailable, static fallback backdrop.', err);
}

/* ORIGINAL INTERACTION 1 — cursor-reactive 3D lighting */
addEventListener('pointermove', (e) => {
  if (!scene3d || reduced) return;
  scene3d.setPointer((e.clientX / window.innerWidth) * 2 - 1,
    -((e.clientY / window.innerHeight) * 2 - 1));
}, { passive: true });

/* ORIGINAL INTERACTION 2 — scroll velocity feeds the silver storm
   inside scene.update via s.velocity. */

/* ORIGINAL INTERACTION 3 — drag-to-inspect (mouse/pen only) */
let dragging = false, lx = 0, ly = 0;
addEventListener('pointerdown', (e) => {
  if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
  if (e.target.closest('a,button,input,.compare,nav')) return;
  dragging = true; lx = e.clientX; ly = e.clientY;
});
addEventListener('pointermove', (e) => {
  if (!dragging || !scene3d || reduced) return;
  scene3d.addSpin(e.clientX - lx, e.clientY - ly);
  lx = e.clientX; ly = e.clientY;
}, { passive: true });
addEventListener('pointerup', () => { dragging = false; });

/* ---------- magnetic buttons ---------- */
if (!reduced && matchMedia('(pointer:fine)').matches) {
  document.querySelectorAll('.btn').forEach((b) => {
    b.addEventListener('mousemove', (e) => {
      const r = b.getBoundingClientRect();
      const x = e.clientX - r.left - r.width / 2;
      const y = e.clientY - r.top - r.height / 2;
      b.style.transform = `translate(${x * 0.18}px,${y * 0.28}px)`;
    });
    b.addEventListener('mouseleave', () => { b.style.transform = ''; });
  });
}

/* ---------- main loop ---------- */
let last = performance.now();
let firstFrames = 0;
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  S.update(dt);
  bar.style.width = (s.progress * 100) + '%';
  lightWords();
  scaleTransStage();
  lightChapters();
  if (scene3d) {
    if (reduced) {
      if (firstFrames === 0) scene3d.update({ smooth: s.smooth, vel: 0, ch: chapterT }, 0, 0, true);
      firstFrames++;
    } else {
      scene3d.update(
        { smooth: s.smooth, vel: s.velocity, ch: chapterT },
        dt, now / 1000, false
      );
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

addEventListener('resize', () => { if (scene3d) scene3d.resize(); });
