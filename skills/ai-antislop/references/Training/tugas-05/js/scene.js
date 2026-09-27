/* scene.js — wisteria-hill DIORAMA (Three.js r160, vendored).
   Procedural stylized diorama (Expeditione/Bruno school): mound +
   wisteria tree (instanced foliage) + clickable stone lantern +
   stone path + petals + fireflies + stars. Scroll = cinematic orbit,
   drag = inspect, click lantern = dusk/dawn toggle. */
import * as THREE from 'three';

const INK = 0x14101d;
const LEAF = [0xf9a8d4, 0xfde9d3, 0xc084fc, 0xe9d5ff];
const FOL = [0x8b5fbf, 0xa87fd4, 0xc9a3e8, 0xf0c6dd, 0x7a4fb0];

function lerp(a, b, t) { return a + (b - a) * t; }
function clamp01(v) { return Math.min(1, Math.max(0, v)); }

/* soft petal sprite */
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

const _c = new THREE.Color();

export function initScene(canvas, opts = {}) {
  const isMobile = opts.mobile || window.innerWidth < 760;
  const PETALS = isMobile ? 160 : 320;
  const FLIES = isMobile ? 30 : 60;
  const DPR = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 1.75);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(DPR);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(INK, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(INK, 0.035);
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 120);

  /* ----- lights ----- */
  scene.add(new THREE.AmbientLight(0x4a4468, 1.3));
  const moon = new THREE.DirectionalLight(0xb8c4e8, 1.0);
  moon.position.set(-5, 8, 4);
  scene.add(moon);
  const cursor = new THREE.PointLight(0xc084fc, 26, 24, 2);
  cursor.position.set(0, 2, 5);
  scene.add(cursor);

  /* ----- diorama group (right side on desktop) ----- */
  const dio = new THREE.Group();
  dio.position.x = isMobile ? 0 : 2.7;
  scene.add(dio);

  // platform + mound
  const plat = new THREE.Mesh(
    new THREE.CylinderGeometry(3.4, 3.7, 0.7, 28),
    new THREE.MeshStandardMaterial({ color: 0x2b2138, roughness: 0.9 })
  );
  plat.position.y = -0.35;
  dio.add(plat);
  const mound = new THREE.Mesh(
    new THREE.SphereGeometry(2.5, 24, 16),
    new THREE.MeshStandardMaterial({ color: 0x3d5233, roughness: 1 })
  );
  mound.scale.set(1, 0.32, 1);
  mound.position.y = 0.1;
  dio.add(mound);

  // wisteria tree: trunk + instanced foliage blobs
  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.3, 2.4, 10),
    new THREE.MeshStandardMaterial({ color: 0x5a4632, roughness: 1 })
  );
  trunk.position.set(-0.9, 1.2, 0.2);
  trunk.rotation.z = 0.08;
  dio.add(trunk);
  const FOLN = isMobile ? 40 : 70;
  const fol = new THREE.InstancedMesh(
    new THREE.IcosahedronGeometry(0.34, 0),
    new THREE.MeshStandardMaterial({ roughness: 0.9 }),
    FOLN
  );
  dio.add(fol);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < FOLN; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.random() * 1.5;
    dummy.position.set(
      -0.9 + Math.cos(a) * r,
      2.5 + Math.random() * 1.3 - r * 0.25,
      0.2 + Math.sin(a) * r
    );
    const s = 0.6 + Math.random() * 0.9;
    dummy.scale.setScalar(s);
    dummy.rotation.set(Math.random() * 3, Math.random() * 3, 0);
    dummy.updateMatrix();
    fol.setMatrixAt(i, dummy.matrix);
    fol.setColorAt(i, _c.setHex(FOL[i % FOL.length]));
  }
  fol.instanceColor.needsUpdate = true;

  // stone lantern (CLICKABLE): base + pillar + light box + roof
  const lantern = new THREE.Group();
  const stoneMat = new THREE.MeshStandardMaterial({ color: 0x6b6570, roughness: 0.95 });
  const add = (geo, y) => { const m = new THREE.Mesh(geo, stoneMat); m.position.y = y; lantern.add(m); return m; };
  add(new THREE.BoxGeometry(0.55, 0.16, 0.55), 0.08);
  add(new THREE.BoxGeometry(0.24, 0.7, 0.24), 0.5);
  const lampMat = new THREE.MeshStandardMaterial({
    color: 0x443a30, emissive: 0xffb066, emissiveIntensity: 2.2, roughness: 0.6,
  });
  const lampBox = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.34, 0.42), lampMat);
  lampBox.position.y = 1.02;
  lantern.add(lampBox);
  const roof = new THREE.Mesh(new THREE.ConeGeometry(0.48, 0.32, 4), stoneMat);
  roof.position.y = 1.35;
  roof.rotation.y = Math.PI / 4;
  lantern.add(roof);
  lantern.position.set(1.5, 0.55, 1.1);
  dio.add(lantern);
  const lampLight = new THREE.PointLight(0xffb066, 36, 11, 2);
  lampLight.position.set(1.5, 1.7, 1.1);
  dio.add(lampLight);

  // stone path
  const pathMat = new THREE.MeshStandardMaterial({ color: 0x59536b, roughness: 1 });
  [[0.4, 2.6], [0.9, 2.0], [0.2, 1.4], [0.7, 0.8]].forEach(([x, z]) => {
    const st = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.38, 0.09, 10), pathMat);
    st.position.set(x, 0.72, z);
    dio.add(st);
  });

  // petals (soft, pink/cream)
  const petals = new THREE.InstancedMesh(
    new THREE.PlaneGeometry(0.12, 0.16),
    new THREE.MeshBasicMaterial({
      map: petalTexture(), side: THREE.DoubleSide, transparent: true,
      depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.85,
    }),
    PETALS
  );
  scene.add(petals); // world-space: drifts across the whole view
  const prm = [];
  for (let i = 0; i < PETALS; i++) {
    prm.push({
      x: (Math.random() - 0.5) * 16, y: -3 + Math.random() * 10, z: -2 + Math.random() * 8,
      fall: 0.2 + Math.random() * 0.45,
      sway: 0.5 + Math.random(), ph: Math.random() * Math.PI * 2,
      rx: Math.random() * 3, rs: (Math.random() - 0.5) * 2,
    });
    petals.setColorAt(i, _c.setHex(LEAF[i % LEAF.length]));
  }
  petals.instanceColor.needsUpdate = true;

  // fireflies (warm drifters near the lantern)
  const flyGeo = new THREE.BufferGeometry();
  const flyPos = new Float32Array(FLIES * 3);
  const flySeed = [];
  for (let i = 0; i < FLIES; i++) {
    flySeed.push({ a: Math.random() * Math.PI * 2, r: 0.8 + Math.random() * 2.2, h: 0.8 + Math.random() * 2, sp: 0.3 + Math.random() * 0.7 });
  }
  flyGeo.setAttribute('position', new THREE.BufferAttribute(flyPos, 3));
  const flies = new THREE.Points(flyGeo, new THREE.PointsMaterial({
    color: 0xffcf87, size: 0.09, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
  }));
  scene.add(flies);

  // stars
  const starN = isMobile ? 200 : 450;
  const sp = new Float32Array(starN * 3);
  for (let i = 0; i < starN; i++) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(22 + Math.random() * 22);
    sp.set([v.x, v.y, v.z], i * 3);
  }
  const starGeo = new THREE.BufferGeometry();
  starGeo.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({
    color: 0x9d94c9, size: 0.07, transparent: true, opacity: 0.75,
  })));

  /* ----- interaction state ----- */
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  const orbit = { az: 0, tAz: 0 };          // drag-orbit inspection (stays)
  const dusk = { v: 1, t: 1 };              // lantern on/off blend (1=lit dusk)
  const lookAt = new THREE.Vector3();
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  /* click test against the lantern (called from UI, NDC coords) */
  function clickLantern(nx, ny) {
    ndc.set(nx, ny);
    ray.setFromCamera(ndc, camera);
    return ray.intersectObjects(lantern.children, false).length > 0;
  }

  /* sc = { smooth, vel, spin (0..1 in interactive band) } */
  function update(sc, dt, t, frozen) {
    pointer.x = lerp(pointer.x, pointer.tx, 1 - Math.pow(0.001, dt));
    pointer.y = lerp(pointer.y, pointer.ty, 1 - Math.pow(0.001, dt));
    cursor.position.x = lerp(cursor.position.x, pointer.x * 5, 0.06);
    cursor.position.y = lerp(cursor.position.y, 1.5 + pointer.y * 3, 0.06);

    // dusk/dawn blend follows toggle target
    dusk.v = lerp(dusk.v, dusk.t, 1 - Math.pow(0.01, dt));
    lampMat.emissiveIntensity = 0.15 + dusk.v * 2.1;
    lampLight.intensity = 4 + dusk.v * 34;
    moon.intensity = 1.5 - dusk.v * 0.6;
    moon.color.setHex(dusk.v > 0.5 ? 0xb8c4e8 : 0xffd9a8);

    if (frozen) { renderer.render(scene, camera); return; }

    // camera: slow scroll orbit + dolly + drag inspection offset
    orbit.az = lerp(orbit.az, orbit.tAz, 1 - Math.pow(0.005, dt));
    const az = sc.smooth * 1.1 + orbit.az;
    const dist = 10.2 - sc.smooth * 1.6 - sc.spin * 1.2;
    camera.position.set(Math.sin(az) * dist, 2.6 - sc.smooth * 0.7, Math.cos(az) * dist);
    lookAt.set(isMobile ? 0 : 1.1, 1.0, 0);
    camera.lookAt(lookAt);

    // lantern flicker (alive when lit)
    lampLight.intensity *= 1 + Math.sin(t * 9.3) * 0.03 * dusk.v;

    // petals fall + scroll gust
    const boost = 1 + Math.min(3, Math.abs(sc.vel) * 0.05);
    for (let i = 0; i < PETALS; i++) {
      const P = prm[i];
      P.y -= P.fall * boost * dt;
      if (P.y < -3.2) { P.y = 7; P.x = (Math.random() - 0.5) * 16; }
      P.rx += P.rs * dt;
      dummy.position.set(P.x + Math.sin(t * P.sway + P.ph) * 0.5, P.y, P.z);
      dummy.rotation.set(P.rx, P.rx * 0.7, 0);
      dummy.updateMatrix();
      petals.setMatrixAt(i, dummy.matrix);
    }
    petals.instanceMatrix.needsUpdate = true;

    // fireflies orbit the lantern
    const arr = flyGeo.attributes.position.array;
    for (let i = 0; i < FLIES; i++) {
      const F = flySeed[i];
      const a = F.a + t * F.sp * 0.4;
      arr[i * 3] = 2.7 + Math.cos(a) * F.r;
      arr[i * 3 + 1] = F.h + Math.sin(t * F.sp + i) * 0.35;
      arr[i * 3 + 2] = 1.1 + Math.sin(a) * F.r;
    }
    flyGeo.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
  }

  return {
    update, resize, clickLantern,
    setPointer(nx, ny) { pointer.tx = nx; pointer.ty = ny; },
    addOrbit(dx) { orbit.tAz += dx * 0.005; },
    toggleDusk() { dusk.t = dusk.t > 0.5 ? 0 : 1; return dusk.t > 0.5; },
    stats() { return { petals: PETALS, calls: renderer.info.render.calls }; },
  };
}
