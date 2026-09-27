/* scene.js — REAL 3D scene (Three.js r160, vendored locally).
   REVERT verdict owner: the "HER TURN" portrait planes looked ugly —
   back to the ambient core + soft petals version (REBLOOM), which the
   owner preferred. Warm palette retained. */
import * as THREE from 'three';

const PALETTE = {
  ink: 0x0b1220,
  wisteria: 0xc084fc,
  blush: 0xf9a8d4,
  cream: 0xfde9d3,
};
const PETAL_COLORS = [0xe8edf5, 0x9fb4d8, 0xf4ede0, 0xe8b64c];

/* camera keyframes over global page progress */
const CAM_STOPS = [
  { p: 0.0, dist: 10.5, y: 1.4, az: 0.0, look: 0.1 },
  { p: 0.22, dist: 8.6, y: 0.7, az: 0.7, look: 0.1 },
  { p: 0.42, dist: 6.0, y: 0.2, az: 1.5, look: 0.0 },
  { p: 0.62, dist: 4.8, y: -0.3, az: 2.5, look: -0.1 },
  { p: 0.82, dist: 7.6, y: 0.9, az: 3.5, look: 0.1 },
  { p: 1.0, dist: 11.5, y: 1.8, az: 4.4, look: 0.2 },
];
/* light color chapters (paired: purple point, pink point) */
const LIGHT_STOPS = [
  { p: 0.0, a: 0x5a7ab8, b: 0x8a9ac0 },
  { p: 0.4, a: 0x9fb4d8, b: 0xe8b64c },
  { p: 0.7, a: 0xc9d4e8, b: 0xf2c078 },
  { p: 1.0, a: 0x5a7ab8, b: 0x8a9ac0 },
];

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp01(v) { return Math.min(1, Math.max(0, v)); }

/* sample a keyframed track [{p, ...keys}] at progress p */
function sampleTrack(stops, p, keys) {
  let i = 0;
  while (i < stops.length - 2 && p > stops[i + 1].p) i++;
  const A = stops[i], B = stops[i + 1];
  const t = clamp01((p - A.p) / Math.max(1e-5, B.p - A.p));
  const s = t * t * (3 - 2 * t); // smoothstep between stops
  const out = {};
  keys.forEach((k) => { out[k] = lerp(A[k], B[k], s); });
  return out;
}

/* soft elliptical petal sprite — no confetti look */
function petalTexture() {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 2, 32, 32, 30);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.45, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.beginPath();
  g.ellipse(32, 32, 13, 23, 0.5, 0, Math.PI * 2);
  g.fill();
  return new THREE.CanvasTexture(c);
}

const _ca = new THREE.Color(), _cb = new THREE.Color();

export function initScene(canvas, opts = {}) {
  const isMobile = opts.mobile || window.innerWidth < 760;
  const PETALS = isMobile ? 220 : 600;
  const DPR = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 1.75);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(DPR);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(PALETTE.ink, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(PALETTE.ink, 0.042);

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 120);

  /* ----- lights ----- */
  scene.add(new THREE.AmbientLight(0x3a4662, 1.4));
  const key = new THREE.DirectionalLight(0xfde9d3, 1.6);
  key.position.set(4, 6, 6);
  scene.add(key);
  const ptA = new THREE.PointLight(0x5a7ab8, 90, 30, 2);
  ptA.position.set(-4.5, 2.5, 3.5);
  scene.add(ptA);
  const ptB = new THREE.PointLight(0x8a9ac0, 60, 30, 2);
  ptB.position.set(4.5, -1.5, 2.5);
  scene.add(ptB);

  /* ----- core artifact (centerpiece, supporting the face hero) ----- */
  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.95, 1),
    new THREE.MeshPhysicalMaterial({
      color: 0x8a93a8, metalness: 0.9, roughness: 0.3,
      clearcoat: 1.0, clearcoatRoughness: 0.25,
      emissive: 0x4a5a78, emissiveIntensity: 0.25,
    })
  );
  scene.add(core);
  const shell = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.0, 1),
    new THREE.MeshBasicMaterial({ color: PALETTE.wisteria, wireframe: true, transparent: true, opacity: 0.14 })
  );
  scene.add(shell);

  /* ----- halo rings ----- */
  const rings = [];
  [[2.25, 0.022, 0.5, 0.5], [2.95, 0.014, 0.32, -0.35]].forEach(([r, tube, op, tilt]) => {
    const m = new THREE.Mesh(
      new THREE.TorusGeometry(r, tube, 12, 140),
      new THREE.MeshBasicMaterial({ color: PALETTE.blush, transparent: true, opacity: op })
    );
    m.rotation.x = Math.PI / 2 + tilt;
    scene.add(m);
    rings.push(m);
  });

  /* ----- petal field (instanced, soft-textured, additive glow) ----- */
  const petalGeo = new THREE.PlaneGeometry(0.16, 0.22);
  const petalMat = new THREE.MeshBasicMaterial({
    map: petalTexture(), side: THREE.DoubleSide,
    transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, opacity: 0.85,
  });
  const petals = new THREE.InstancedMesh(petalGeo, petalMat, PETALS);
  scene.add(petals);
  const dummy = new THREE.Object3D();
  const prm = [];
  for (let i = 0; i < PETALS; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = 2.8 + Math.random() * 5.2;
    prm.push({
      x: Math.cos(a) * r, y: -4.5 + Math.random() * 11, z: Math.sin(a) * r,
      fall: 0.25 + Math.random() * 0.55,
      sway: 0.4 + Math.random() * 0.9, ph: Math.random() * Math.PI * 2,
      rx: Math.random() * Math.PI, ry: Math.random() * Math.PI,
      rs: (Math.random() - 0.5) * 2.2,
    });
    petals.setColorAt(i, _ca.setHex(PETAL_COLORS[i % PETAL_COLORS.length]));
  }
  petals.instanceColor.needsUpdate = true;

  /* ----- starfield ----- */
  const starN = isMobile ? 220 : 500;
  const pos = new Float32Array(starN * 3);
  for (let i = 0; i < starN; i++) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(20 + Math.random() * 22);
    pos.set([v.x, v.y, v.z], i * 3);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
    color: 0x9d94c9, size: 0.07, transparent: true, opacity: 0.8, sizeAttenuation: true,
  }));
  scene.add(stars);

  /* ----- interaction state ----- */
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };   // cursor-reactive lighting
  const spin = { x: 0, y: 0 };                     // drag impulse, decays
  const lookAt = new THREE.Vector3();

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  /* sc = { smooth (0..1 page), vel (px/frame smoothed), ch (0..1 in 3D chapter) } */
  function update(sc, dt, t, frozen) {
    // cursor light follow (lerped = spring-ish)
    pointer.x = lerp(pointer.x, pointer.tx, 1 - Math.pow(0.001, dt));
    pointer.y = lerp(pointer.y, pointer.ty, 1 - Math.pow(0.001, dt));
    ptA.position.x = lerp(ptA.position.x, -4.5 + pointer.x * 3.2, 0.06);
    ptA.position.y = lerp(ptA.position.y, 2.5 + pointer.y * 2.2, 0.06);
    ptB.position.x = lerp(ptB.position.x, 4.5 + pointer.x * 2.4, 0.06);

    if (frozen) { renderer.render(scene, camera); return; }

    // spin impulse decay (drag-to-inspect springs back to scroll pose)
    spin.x *= Math.pow(0.02, dt);
    spin.y *= Math.pow(0.02, dt);

    // camera dolly along keyframes
    const cam = sampleTrack(CAM_STOPS, sc.smooth, ['dist', 'y', 'az', 'look']);
    camera.position.set(
      Math.sin(cam.az) * cam.dist,
      cam.y,
      Math.cos(cam.az) * cam.dist
    );
    lookAt.set(0, cam.look, 0);
    camera.lookAt(lookAt);

    // core: time spin + scroll-chapter turn + drag impulse
    core.rotation.y = t * 0.16 + sc.ch * Math.PI * 2.4 + spin.y;
    core.rotation.x = Math.sin(t * 0.22) * 0.18 + spin.x;
    shell.rotation.y = -t * 0.1 + sc.ch * Math.PI * 1.6 + spin.y * 1.2;
    shell.rotation.z = t * 0.06;
    const breathe = 1 + Math.sin(t * 0.9) * 0.02 + sc.ch * 0.14;
    core.scale.setScalar(breathe);
    core.material.emissiveIntensity = 0.25 + sc.ch * 0.45 + Math.abs(sc.vel) * 0.004;

    rings[0].rotation.z = t * 0.12;
    rings[1].rotation.z = -t * 0.09;

    // light color chapters (manual THREE.Color lerp between stops)
    const li = (() => {
      let i = 0;
      while (i < LIGHT_STOPS.length - 2 && sc.smooth > LIGHT_STOPS[i + 1].p) i++;
      const A = LIGHT_STOPS[i], B = LIGHT_STOPS[i + 1];
      const k = clamp01((sc.smooth - A.p) / Math.max(1e-5, B.p - A.p));
      return { A, B, k: k * k * (3 - 2 * k) };
    })();
    ptA.color.copy(_ca.setHex(li.A.a)).lerp(_cb.setHex(li.B.a), li.k);
    ptB.color.copy(_ca.setHex(li.A.b)).lerp(_cb.setHex(li.B.b), li.k);

    // petals: fall + sway + scroll-velocity storm
    const boost = 1 + Math.min(3.2, Math.abs(sc.vel) * 0.05);
    for (let i = 0; i < PETALS; i++) {
      const P = prm[i];
      P.y -= P.fall * boost * dt;
      if (P.y < -4.5) { P.y = 6.5; P.x = (Math.random() - 0.5) * 12; }
      P.rx += P.rs * dt * boost;
      P.ry += P.rs * 0.7 * dt;
      dummy.position.set(
        P.x + Math.sin(t * P.sway + P.ph) * 0.5 * boost,
        P.y, P.z
      );
      dummy.rotation.set(P.rx, P.ry, 0);
      dummy.updateMatrix();
      petals.setMatrixAt(i, dummy.matrix);
    }
    petals.instanceMatrix.needsUpdate = true;

    stars.rotation.y = t * 0.008;
    renderer.render(scene, camera);
  }

  function stats() {
    return { petals: PETALS, calls: renderer.info.render.calls, tris: renderer.info.render.triangles };
  }

  return {
    update, resize, stats,
    setPointer(nx, ny) { pointer.tx = nx; pointer.ty = ny; },
    addSpin(dx, dy) { spin.y += dx * 0.004; spin.x += dy * 0.004; },
  };
}
