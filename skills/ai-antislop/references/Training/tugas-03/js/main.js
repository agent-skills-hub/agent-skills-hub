/* main.js — UI + interaction wiring. Owns: progress bar, hero type,
   marquee, reveals, word scrub, horizontal gallery, chapter lighting,
   lightbox, drag-to-inspect, cursor lights, loop, resize, fallbacks. */
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
// charin keyframes live in the page (hero section owns its entrance)
const st = document.createElement('style');
st.textContent = '@keyframes charin{from{opacity:0;transform:translateY(70px) rotate(5deg)}to{opacity:1;transform:none}}';
document.head.appendChild(st);

/* ---------- marquee seamless loop ---------- */
const track = document.getElementById('mtrack');
if (track) track.innerHTML += track.innerHTML;

/* ---------- reveals ---------- */
S.observeReveals();

/* ---------- word scrub (story paragraph lights up with scroll) ---------- */
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

/* ---------- horizontal gallery (vertical scroll drives X) ---------- */
const hwrap = document.getElementById('hwrap');
const htrack = document.getElementById('htrack');
function slideGallery() {
  if (!hwrap || reduced) return;
  const p = S.through(hwrap); // 0..1 across the tall section
  // ease the ends so the strip starts/settles gracefully
  const e = p * p * (3 - 2 * p);
  const maxX = Math.max(0, htrack.scrollWidth - window.innerWidth + window.innerWidth * 0.07);
  htrack.style.transform = `translate3d(${-e * maxX}px,0,0)`;
}

/* ---------- transformation stage: scroll drives scale ---------- */
const transStage = document.getElementById('transStage');
const transImg = document.getElementById('transImg');
function scaleTransStage() {
  if (!transStage || reduced) return;
  const p = S.through(transStage); // 0 entering → 1 leaving
  // dramatic: starts small, blooms past full-bleed scale mid-view
  const sc = 0.82 + Math.sin(Math.min(1, p * 1.15) * Math.PI) * 0.26;
  transImg.style.transform = `scale(${sc.toFixed(4)})`;
}
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

/* ---------- lightbox (gallery inspection) ---------- */
const lb = document.getElementById('lightbox');
const lbImg = document.getElementById('lbImg');
const lbCap = document.getElementById('lbCap');
function openLB(src, cap) {
  lbImg.src = src; lbImg.alt = cap;
  lbCap.textContent = cap;
  lb.classList.add('open');
  document.getElementById('lbClose').focus();
}
function closeLB() { lb.classList.remove('open'); lbImg.removeAttribute('src'); }
document.querySelectorAll('.hcard').forEach((c) => {
  c.addEventListener('click', () => {
    const img = c.querySelector('img');
    openLB(img.src, c.querySelector('.cap').textContent.trim());
  });
});
document.getElementById('lbClose').addEventListener('click', closeLB);
lb.addEventListener('click', (e) => { if (e.target === lb) closeLB(); });
addEventListener('keydown', (e) => { if (e.key === 'Escape' && lb.classList.contains('open')) closeLB(); });

/* ---------- 3D scene (real WebGL, vendored three.js) ---------- */
let scene3d = null;
try {
  scene3d = initScene(document.getElementById('gl'), { mobile });
  const info = scene3d.stats();
  console.info(`[hana] WebGL scene live — petals:${info.petals} calls:${info.calls} tris:${info.tris}`);
} catch (err) {
  document.body.classList.add('no-gl');
  console.warn('[hana] WebGL unavailable, static fallback backdrop.', err);
}

/* ORIGINAL INTERACTION 1 — cursor-reactive 3D lighting */
addEventListener('pointermove', (e) => {
  if (!scene3d || reduced) return;
  scene3d.setPointer((e.clientX / window.innerWidth) * 2 - 1,
    -((e.clientY / window.innerHeight) * 2 - 1));
}, { passive: true });

/* ORIGINAL INTERACTION 2 — scroll velocity already feeds the petal
   storm inside scene.update via s.velocity. Nothing extra needed here. */

/* ORIGINAL INTERACTION 3 — drag-to-inspect: drag empty space to spin
   the artifact; it springs back to the scroll pose on release.
   Mouse/pen only — touch stays 100% scroll. */
let dragging = false, lx = 0, ly = 0;
addEventListener('pointerdown', (e) => {
  if (e.pointerType !== 'mouse' && e.pointerType !== 'pen') return;
  if (e.target.closest('a,button,.hcard,nav')) return;
  dragging = true; lx = e.clientX; ly = e.clientY;
});
addEventListener('pointermove', (e) => {
  if (!dragging || !scene3d || reduced) return;
  scene3d.addSpin(e.clientX - lx, e.clientY - ly);
  lx = e.clientX; ly = e.clientY;
}, { passive: true });
addEventListener('pointerup', () => { dragging = false; });

/* ---------- magnetic buttons (micro-interaction) ---------- */
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
  slideGallery();
  scaleTransStage();
  lightChapters();
  if (scene3d) {
    if (reduced) {
      // one static frame only — no animation loop cost
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
