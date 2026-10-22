// Kunstwerk-dashboard: the kunstwerk's local frame drawn as is (x east, y north, z up, metres; tools/dashboard/geo.py).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const $ = (s) => document.querySelector(s);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${await r.text()}`);
  return r;
};
const postJSON = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body || {}) });
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function log(line, cls) {
  const el = $('#log');
  const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 4;
  el.append(document.createTextNode(line + '\n'));
  if (atEnd) el.scrollTop = el.scrollHeight;
}

// ---- layers --------------------------------------------------------------------------------------------------------
const LAYERS = [
  { key: 'mesh', name: 'Kunstwerk (export)', color: '#9e9e9e', on: true },
  { key: 'ground', name: 'Luchtfoto + AHN', color: '#6d8f5a', on: true },
  { key: 'ours', name: 'Onze borden & VRI\'s', color: '#4fc3f7', on: true },
  { key: 'direction', name: 'Wegwijzers', color: '#ce93d8', on: true },
  { key: 'heads', name: 'Signaalkoppen (export)', color: '#ff5252', on: true },
  { key: 'furniture', name: 'Meubilair in mesh', color: '#bdbdbd', on: false },
  { key: 'ndw', name: 'NDW-borden', color: '#81c784', on: true },
  { key: 'osm_signals', name: 'OSM-knopen', color: '#ffd54f', on: false },
  { key: 'osm_ways', name: 'OSM-wegen', color: '#ffa726', on: false },
  { key: 'approaches', name: 'Aanrijrichtingen', color: '#90a4ae', on: true },
  { key: 'sv', name: 'Street View: detecties', color: '#e040fb', on: true },
  { key: 'sv_shots', name: 'Street View: camera\'s', color: '#80deea', on: false },
];
const VERDICT = {
  ok: ['#66bb6a', 'ok'], moved: ['#ffa726', 'verplaatst'], missing_in_ours: ['#ef5350', 'mist bij ons'],
  extra_in_ours: ['#ab47bc', 'extra bij ons'], wrong_facing: ['#ffee58', 'verkeerde richting'],
  wrong_code: ['#26c6da', 'verkeerde code'],
};
const MAT = {
  bne_grass: 0x557d3f, sidewalk_concrete: 0xa9a9a2, bne_cycle_red: 0x8e3b33, road_asphalt_2lane: 0x3b3c40,
  bne_concrete_dark: 0x707070, curbs: 0xbdbdbd, bne_truss_steel: 0x5b6070, bne_paint_white: 0xf2f2f2,
  bne_tunnel_tiles: 0xd9d8cf, bne_tunnel_light: 0xfff4c0, bne_rail_steel: 0x8a8f99, bne_steel_black: 0x222326,
  bne_signs: 0x8d96a3, bne_sidewalk_brick: 0x9b7c69, italy_cobblestone: 0x7f705f, bne_tunnel_soil: 0x6b5a45,
  bne_bridge_concrete: 0xb2ada4,
};

const state = { info: null, objects: null, ground: null, sv: null, byId: new Map(), selected: null, edit: null,
  showLabels: true, showArrows: true };
const groups = {};

// ---- scene ---------------------------------------------------------------------------------------------------------
const view = $('#view');
const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = 'absolute';
labels.domElement.style.top = '0';
labels.domElement.style.pointerEvents = 'none';
view.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1e24);
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const camera = new THREE.PerspectiveCamera(50, 1, 0.2, 5000);
camera.up.set(0, 0, 1);
camera.position.set(-40, -170, 170);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 30, 4);
orbit.maxPolarAngle = Math.PI * 0.495;
orbit.update();
scene.add(new THREE.HemisphereLight(0xdde6ff, 0x3a3020, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-120, -80, 200);
scene.add(sun);
for (const L of LAYERS) {
  const g = new THREE.Group();
  g.name = L.key;
  g.visible = L.on;
  groups[L.key] = g;
  scene.add(g);
}
const editHelpers = new THREE.Group();
scene.add(editHelpers);

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.setSpace('world');
gizmo.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value; });
gizmo.addEventListener('objectChange', () => onGizmoChange());
scene.add(gizmo);

function resize() {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h);
  labels.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(view);
resize();
renderer.setAnimationLoop(() => {
  orbit.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
});

// ---- ground height (the AHN grid the server sends with the photo) ----------------------------------------------------
function groundZ(x, y) {
  const g = state.ground;
  if (!g || !g.present) return 0;
  const [x0, y0, x1, y1] = g.box;
  const fx = (x - x0) / (x1 - x0) * g.nx, fy = (y1 - y) / (y1 - y0) * g.ny;
  const i = Math.min(Math.max(Math.round(fy), 0), g.ny), j = Math.min(Math.max(Math.round(fx), 0), g.nx);
  return g.z[i * (g.nx + 1) + j];
}

function clear(g) {
  for (const c of [...g.children]) {
    g.remove(c);
    c.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.element) o.element.remove();
    });
  }
}

// ---- kunstwerk mesh --------------------------------------------------------------------------------------------------
async function loadMesh() {
  clear(groups.mesh);
  clear(groups.heads);
  let buf;
  try {
    buf = await (await api('/api/mesh.bin')).arrayBuffer();
  } catch (e) {
    log('mesh: ' + e.message);
    return;
  }
  const hl = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const pos = new Float32Array(buf, 4 + hl, head.nverts * 3);
  const idx = new Uint32Array(buf, 4 + hl + head.nverts * 12, head.ntris * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  const mats = [];
  head.groups.forEach((g, k) => {
    geo.addGroup(g.start, g.count, k);
    const col = MAT[g.material] ?? new THREE.Color().setHSL((k * 0.137) % 1, 0.3, 0.55).getHex();
    mats.push(new THREE.MeshLambertMaterial({ color: col, side: THREE.DoubleSide,
      emissive: g.material === 'bne_tunnel_light' ? 0x554e30 : 0x000000 }));
  });
  const m = new THREE.Mesh(geo, mats);
  m.name = 'kunstwerk';
  groups.mesh.add(m);
  // its overhead wires
  const wmat = new THREE.LineBasicMaterial({ color: 0x111111 });
  for (const w of head.wires || []) {
    for (const ln of (Array.isArray(w[0][0]) ? w : [w])) {
      const g = new THREE.BufferGeometry().setFromPoints(ln.map((p) => new THREE.Vector3(p[0], p[1], p[2])));
      groups.mesh.add(new THREE.Line(g, wmat));
    }
  }
  $('#hud').textContent = `${head.name}: ${head.nverts.toLocaleString('nl')} hoekpunten, ` +
    `${head.ntris.toLocaleString('nl')} driehoeken · export ${new Date(head.mtime / 1e6).toLocaleString('nl')}`;
  setCount('mesh', head.ntris);
}

// ---- ground photo ----------------------------------------------------------------------------------------------------
async function loadGround() {
  state.ground = await (await api('/api/ground')).json();
  clear(groups.ground);
  const g = state.ground;
  if (!g.present) return;
  const [x0, y0, x1, y1] = g.box;
  const geo = new THREE.PlaneGeometry(x1 - x0, y1 - y0, g.nx, g.ny);
  const p = geo.attributes.position;
  for (let k = 0; k < p.count; k++) p.setZ(k, g.z[k] - 0.25);
  geo.translate((x0 + x1) / 2, (y0 + y1) / 2, 0);
  const tex = new THREE.TextureLoader().load('/api/ground/photo.jpg');
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const mat = new THREE.MeshBasicMaterial({ map: tex, polygonOffset: true, polygonOffsetFactor: 2,
    polygonOffsetUnits: 2 });
  groups.ground.add(new THREE.Mesh(geo, mat));
  setCount('ground', '');
}

// ---- markers ---------------------------------------------------------------------------------------------------------
const poleGeo = new THREE.CylinderGeometry(0.06, 0.06, 1, 8).rotateX(Math.PI / 2).translate(0, 0, 0.5);
const plateGeo = new THREE.BoxGeometry(0.7, 0.06, 0.7);
const discGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.06, 20);
const headGeo = new THREE.BoxGeometry(0.36, 0.25, 1.0);
const dotGeo = new THREE.SphereGeometry(0.45, 14, 10);
const matCache = new Map();
function mat(color, opts = {}) {
  const key = color + JSON.stringify(opts);
  if (!matCache.has(key)) matCache.set(key, new THREE.MeshLambertMaterial({ color, ...opts }));
  return matCache.get(key);
}

function label(text, color, cls = '') {
  const el = document.createElement('div');
  el.className = 'lbl ' + cls;
  el.textContent = text;
  if (color) el.style.setProperty('--c', color);
  const o = new CSS2DObject(el);
  o.userData.isLabel = true;
  o.visible = state.showLabels;
  return o;
}

function arrow(color, len = 2.2) {
  const a = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), len, color, 0.6, 0.4);
  a.userData.isArrow = true;
  a.visible = state.showArrows;
  return a;
}

// rotation about z that turns a marker's +y (its face) to the facing [fx, fy]; heading = compass degrees
const rotOf = (f) => Math.atan2(-f[0], f[1]);
const headingOfRot = (r) => ((-r * 180 / Math.PI) % 360 + 360) % 360;

function marker(o, color) {
  const g = new THREE.Group();
  g.position.set(o.at[0], o.at[1], o.z);
  if (o.facing) g.rotation.z = rotOf(o.facing);
  const h = o.kind === 'traffic_light' ? (o.props.mast ? 6.3 : 3.3) : o.kind === 'signal_head' ? 0 : 2.6;
  const pick = [];
  const add = (mesh) => { mesh.userData.obj = o; pick.push(mesh); g.add(mesh); return mesh; };
  const removed = o.props && o.props.removed;
  const m = mat(color, removed ? { transparent: true, opacity: 0.3 } : {});
  if (o.kind === 'signal_head') {
    g.position.z = o.z;                                     // (its head's own height)
    add(new THREE.Mesh(headGeo, m));
  } else if (['osm_signals', 'approaches', 'furniture'].includes(o.layer)) {
    add(new THREE.Mesh(dotGeo, m)).position.z = 0.6;
  } else {
    const pole = add(new THREE.Mesh(poleGeo, mat(0x9aa0aa, removed ? { transparent: true, opacity: 0.3 } : {})));
    pole.scale.z = h;
    const plate = add(new THREE.Mesh(o.kind === 'traffic_light' ? headGeo : (o.layer === 'ndw' ? discGeo : plateGeo), m));
    if (plate.geometry === discGeo) plate.rotation.x = Math.PI / 2;
    plate.position.set(0, 0.12, h - (o.kind === 'traffic_light' ? 0.8 : 0.35));
  }
  if (o.facing) {
    const a = arrow(color);
    a.position.z = o.kind === 'signal_head' ? 0.7 : h + 0.4;
    g.add(a);
  }
  if (!['heads', 'furniture'].includes(o.layer)) {
    const lb = label(o.label, color);
    lb.position.z = h + 1.0;
    g.add(lb);
  }
  g.userData = { obj: o, pick };
  return g;
}

// a label shows with its layer (CSS2DRenderer looks at the label's own visible flag only)
function refreshLabels() {
  for (const L of LAYERS) {
    const on = state.showLabels && groups[L.key].visible;
    groups[L.key].traverse((o) => { if (o.userData.isLabel) o.visible = on; });
  }
}

function setCount(key, n) {
  const el = document.querySelector(`.layer[data-key="${key}"] .n`);
  if (el) el.textContent = n;
}

function buildLayer(key, objs, color) {
  clear(groups[key]);
  for (const o of objs) {
    const g = marker(o, key === 'ours' && o.kind === 'traffic_light' ? '#ffb74d' : color);
    groups[key].add(g);
    state.byId.set(o.id, { obj: o, group: g });
  }
  setCount(key, objs.length);
}

// moved objects: a thin line from their place before the override
function buildEditHelpers() {
  clear(editHelpers);
  const lm = new THREE.LineDashedMaterial({ color: 0xffb74d, dashSize: 0.5, gapSize: 0.3 });
  for (const o of [...state.objects.layers.ours, ...state.objects.layers.direction]) {
    const b = o.props.base_at;
    if (!b || (Math.abs(b[0] - o.at[0]) < 0.01 && Math.abs(b[1] - o.at[1]) < 0.01)) continue;
    const z = o.z + 0.3;
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(b[0], b[1], groundZ(b[0], b[1]) + 0.3),
      new THREE.Vector3(o.at[0], o.at[1], z)]), lm);
    l.computeLineDistances();
    editHelpers.add(l);
  }
}

async function loadObjects() {
  state.objects = await (await api('/api/objects')).json();
  for (const [k, v] of [...state.byId]) if (!k.startsWith('sv:')) state.byId.delete(k);
  const L = Object.fromEntries(LAYERS.map((l) => [l.key, l.color]));
  for (const key of ['ours', 'direction', 'heads', 'furniture', 'ndw', 'osm_signals']) {
    buildLayer(key, state.objects.layers[key] || [], L[key]);
  }
  // approaches: their stop line and an arrow the way in
  clear(groups.approaches);
  for (const o of state.objects.layers.approaches) {
    const g = marker(o, L.approaches);
    const [l, r] = [o.props.left, o.props.right];
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(l[0] - o.at[0], l[1] - o.at[1], 0.3), new THREE.Vector3(r[0] - o.at[0], r[1] - o.at[1], 0.3)]),
    new THREE.LineBasicMaterial({ color: 0xffffff }));
    line.rotation.z = -g.rotation.z;
    g.add(line);
    groups.approaches.add(g);
    state.byId.set(o.id, { obj: o, group: g });
  }
  setCount('approaches', state.objects.layers.approaches.length);
  for (const m of state.objects.messages || []) log('pijplijn: ' + m);
  buildEditHelpers();
  if (state.sv) buildSV();
  refreshLabels();
}

async function loadWays() {
  const { ways } = await (await api('/api/osm/ways')).json();
  clear(groups.osm_ways);
  const col = { primary: 0xffa726, secondary: 0xffcc80, tertiary: 0xffe0b2, cycleway: 0xef5350, footway: 0xbdbdbd,
    tram: 0x4dd0e1 };
  for (const w of ways) {
    const pts = w.pts.map((p) => new THREE.Vector3(p[0], p[1], groundZ(p[0], p[1]) + 0.4));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: col[w.kind] ?? 0xfff3e0 }));
    groups.osm_ways.add(line);
  }
  setCount('osm_ways', ways.length);
}

// ---- Street View diagnosis -------------------------------------------------------------------------------------------
async function loadSV(place) {
  state.sv = await (await api('/api/sv?place=' + encodeURIComponent(place || ''))).json();
  const s = state.sv;
  $('#sv-info').textContent = s.present ? `${s.objects.length} detecties, ${s.shots.length} camera's`
    : `geen scan in ${state.info.sv_root}\\${s.place}`;
  buildSV();
}

function buildSV() {
  clear(groups.sv);
  clear(groups.sv_shots);
  for (const k of [...state.byId.keys()]) if (k.startsWith('sv:')) state.byId.delete(k);
  const s = state.sv;
  if (!s) return;
  for (const o of s.objects) {
    const color = (VERDICT[o.verdict] || ['#e040fb'])[0];
    const g = new THREE.Group();
    const hz = o.props.height_m != null ? Number(o.props.height_m) : 2.5;
    g.position.set(o.at[0], o.at[1], o.z);
    const dot = new THREE.Mesh(new THREE.OctahedronGeometry(0.55), mat(color, { emissive: color, emissiveIntensity: 0.3 }));
    dot.position.z = hz;
    dot.userData.obj = o;
    g.add(dot);
    const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, hz)]), new THREE.LineBasicMaterial({ color }));
    g.add(stem);
    if (o.facing) {
      const a = arrow(color, 1.8);
      a.setDirection(new THREE.Vector3(o.facing[0], o.facing[1], 0).normalize());
      a.position.z = hz + 0.7;
      g.add(a);
    }
    const lb = label(`${o.label} · ${(VERDICT[o.verdict] || [0, o.verdict])[1]}`, color, 'sv');
    lb.position.z = hz + 1.5;
    g.add(lb);
    g.userData = { obj: o, pick: [dot] };
    groups.sv.add(g);
    state.byId.set(o.id, { obj: o, group: g });
    // the line to what it matched
    const m = o.props.match || {};
    let to = null;
    const hit = (o.match_id && state.byId.get(o.match_id)) || (m.ref != null && state.byId.get(String(m.ref)));
    if (hit) to = new THREE.Vector3(hit.obj.at[0], hit.obj.at[1], hit.obj.z + 2.0);
    else if (o.match_at) to = new THREE.Vector3(o.match_at[0], o.match_at[1], groundZ(...o.match_at) + 2.0);
    if (to) {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(o.at[0], o.at[1], o.z + hz), to]), new THREE.LineBasicMaterial({ color }));
      groups.sv.add(l);
    }
  }
  // camera frusta
  const fm = new THREE.LineBasicMaterial({ color: 0x80deea, transparent: true, opacity: 0.7 });
  for (const sh of s.shots) {
    const g = new THREE.Group();
    g.position.set(sh.at[0], sh.at[1], sh.z);
    const L = 8, hf = Math.tan((sh.fov * Math.PI / 180) / 2), vf = hf * sh.h / Math.max(sh.w, 1);
    const hd = sh.heading * Math.PI / 180, pt = sh.pitch * Math.PI / 180;
    const fwd = new THREE.Vector3(Math.sin(hd) * Math.cos(pt), Math.cos(hd) * Math.cos(pt), Math.sin(pt));
    const right = new THREE.Vector3(Math.cos(hd), -Math.sin(hd), 0);
    const up = new THREE.Vector3().crossVectors(right, fwd);
    const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => fwd.clone().add(right.clone().multiplyScalar(a * hf))
      .add(up.clone().multiplyScalar(b * vf)).multiplyScalar(L));
    const o0 = new THREE.Vector3();
    const segs = [o0, c[0], o0, c[1], o0, c[2], o0, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]];
    g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segs), fm));
    const face = new THREE.Mesh(new THREE.BufferGeometry().setFromPoints([c[0], c[1], c[2], c[0], c[2], c[3]]),
      new THREE.MeshBasicMaterial({ color: 0x80deea, transparent: true, opacity: 0.12, side: THREE.DoubleSide }));
    const cam = new THREE.Mesh(new THREE.SphereGeometry(0.35, 10, 8), mat('#80deea'));
    const shotObj = { id: 'shot:' + sh.id, layer: 'sv_shots', kind: 'shot', label: sh.id, shot: sh, at: sh.at, z: sh.z };
    face.userData.obj = shotObj;
    cam.userData.obj = shotObj;
    g.add(face, cam);
    g.userData = { obj: shotObj, pick: [face, cam] };
    groups.sv_shots.add(g);
    state.byId.set(shotObj.id, { obj: shotObj, group: g });
  }
  setCount('sv', s.objects.length);
  setCount('sv_shots', s.shots.length);
  refreshLabels();
}

// ---- picking ---------------------------------------------------------------------------------------------------------
const ray = new THREE.Raycaster();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4 || gizmo.dragging) return;
  const r = renderer.domElement.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const pickables = [];
  for (const L of LAYERS) {
    if (!groups[L.key].visible) continue;
    groups[L.key].traverse((o) => { if (o.userData && o.userData.obj && o.isMesh) pickables.push(o); });
  }
  const hits = ray.intersectObjects(pickables, false);
  if (hits.length) select(hits[0].object.userData.obj.id);
  else if (!e.shiftKey) select(null);
});
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Escape') select(null);
  if (e.key === 'g' && state.edit) setGizmo('translate');
  if (e.key === 'r' && state.edit) setGizmo('rotate');
  if (e.key === 'f' && state.selected) focusOn(state.selected);
});

function focusOn(id) {
  const e = state.byId.get(id);
  if (!e) return;
  const p = new THREE.Vector3();
  e.group.getWorldPosition(p);
  const d = camera.position.clone().sub(orbit.target);
  orbit.target.copy(p);
  camera.position.copy(p.clone().add(d.setLength(Math.min(d.length(), 45))));
}

// ---- inspector -------------------------------------------------------------------------------------------------------
function propsTable(obj) {
  const rows = [];
  const put = (k, v) => rows.push(`<tr><td>${esc(k)}</td><td>${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</td></tr>`);
  put('id', obj.id);
  put('laag', obj.layer);
  if (obj.at) put('x, y (lokaal)', `${obj.at[0].toFixed(2)}, ${obj.at[1].toFixed(2)}`);
  if (obj.z != null) put('z (NAP)', Number(obj.z).toFixed(2));
  if (obj.heading != null) put('richting', `${obj.heading.toFixed(1)}° (0 = noord)`);
  for (const [k, v] of Object.entries(obj.props || {})) {
    if (['id', 'override', 'base_at', 'base_facing'].includes(k) || v == null || v === '') continue;
    put(k, v);
  }
  return `<table class="props">${rows.join('')}</table>`;
}

function select(id) {
  gizmo.detach();
  if (state.edit) restoreEdited();
  state.edit = null;
  state.selected = id;
  const el = $('#inspector');
  if (!id || !state.byId.has(id)) {
    el.innerHTML = 'Klik een object of camerapositie aan.';
    el.className = 'muted';
    return;
  }
  el.className = '';
  const { obj } = state.byId.get(id);
  let html = `<div><b>${esc(obj.label)}</b> <span class="badge">${esc(obj.kind)}</span>` +
    (obj.props && obj.props.override ? ' <span class="badge ov">override</span>' : '') + '</div>';
  if (obj.layer === 'sv') {
    const v = VERDICT[obj.verdict] || ['#ccc', obj.verdict];
    html += `<div style="margin-top:4px"><span class="badge" style="color:${v[0]}">${esc(v[1])}</span></div>`;
  }
  html += propsTable(obj);
  if (obj.layer !== 'sv_shots') html += viewsHTML(obj);
  if (obj.layer === 'sv' && obj.match_id) {
    html += `<div class="row"><button id="btn-match">Naar match: ${esc(obj.match_id)}</button></div>`;
  }
  if (obj.layer === 'sv_shots') {
    html += shotHTML(obj.shot, true);
  } else if (obj.layer === 'sv' && (obj.props.shots || []).length) {
    html += '<div class="row">' + obj.props.shots.map((s) => `<button data-shot="${esc(s)}">${esc(s)}</button>`).join('') + '</div><div id="shot-view"></div>';
  }
  if (obj.editable) html += editForm(obj);
  html += `<div class="row"><button id="btn-focus">Inzoomen (f)</button></div>`;
  el.innerHTML = html;
  $('#btn-focus').onclick = () => focusOn(id);
  if ($('#btn-match')) $('#btn-match').onclick = () => { if (state.byId.has(obj.match_id)) { select(obj.match_id); focusOn(obj.match_id); } };
  el.querySelectorAll('[data-shot]').forEach((b) => b.onclick = () => {
    const sh = (state.sv.shots || []).find((s) => s.id === b.dataset.shot) || { id: b.dataset.shot, png: true, annotated: true };
    $('#shot-view').innerHTML = shotHTML(sh, true);
    wireShot();
  });
  wireShot();
  wireViews();
  if (obj.editable) wireEdit(obj);
}

// Which screenshots see a point: the shot's camera (at, z = ground + 2.5 m, compass heading, pitch, horizontal
// fov over w px) as a pinhole; the point's pixel from its bearing and elevation. A Street View detection lists
// its own shots first; any other shot counts when the point falls inside its frame within 90 m.
function project(sh, p, z) {
  const dx = p[0] - sh.at[0], dy = p[1] - sh.at[1], dist = Math.hypot(dx, dy);
  if (dist < 1 || dist > 90) return null;
  let yaw = Math.atan2(dx, dy) * 180 / Math.PI - sh.heading;
  yaw = ((yaw + 540) % 360) - 180;
  if (Math.abs(yaw) >= 80) return null;
  const f = (sh.w / 2) / Math.tan(sh.fov * Math.PI / 360);
  const r = yaw * Math.PI / 180, el = Math.atan2((z ?? sh.z) - sh.z, dist) - (sh.pitch || 0) * Math.PI / 180;
  const u = sh.w / 2 + f * Math.tan(r), v = sh.h / 2 - f * Math.tan(el) / Math.cos(r);
  if (u < 0 || u > sh.w || v < 0 || v > sh.h) return null;
  return { u, v, dist };
}

function viewsFor(obj) {
  const shots = (state.sv && state.sv.shots) || [];
  if (!shots.length || !obj.at) return [];
  const own = (obj.layer === 'sv' && obj.props && obj.props.shots) || [];
  const out = [];
  for (const sh of shots) {
    if (!sh.png) continue;
    const pr = project(sh, obj.at, obj.z);
    if (pr) out.push({ sh, ...pr, own: own.includes(sh.id) });
  }
  out.sort((a, b) => (b.own - a.own) || (a.dist - b.dist));
  return out.slice(0, 8);
}

function viewsHTML(obj) {
  const vs = viewsFor(obj);
  if (!vs.length) return (state.sv && state.sv.shots && state.sv.shots.length)
    ? '<div class="muted small" style="margin-top:6px">Geen Street View-opname ziet dit punt.</div>' : '';
  // a wide crop: Street View's camera position is 1–3 m off, so near objects land well away from the mark
  const place = encodeURIComponent(state.sv.place), k = 1.3, cw = 124, ch = 104;
  return '<div class="views-head">Street View (' + vs.length + ')</div><div class="views">' + vs.map((x) => {
    const url = `/api/sv/shot/${place}/${encodeURIComponent(x.sh.id)}.png`;
    const bx = -(x.u * k - cw / 2), by = -(x.v * k - ch / 2);
    return `<div class="crop${x.own ? ' own' : ''}" data-url="${url}" data-u="${x.u}" data-v="${x.v}" data-w="${x.sh.w}" data-h="${x.sh.h}" data-id="${esc(x.sh.id)}"` +
      ` title="${esc(x.sh.id)} · ${x.dist.toFixed(0)} m" style="width:${cw}px;height:${ch}px;background-image:url('${url}');` +
      `background-size:${x.sh.w * k}px ${x.sh.h * k}px;background-position:${bx}px ${by}px"><span>${esc(x.sh.id)} · ${x.dist.toFixed(0)} m</span></div>`;
  }).join('') + '</div>';
}

function wireViews() {
  document.querySelectorAll('#inspector .crop').forEach((c) => c.onclick = () => openShot(c.dataset));
}

function openShot(d) {
  let lb = $('#lightbox');
  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'lightbox';
    document.body.appendChild(lb);
    lb.onclick = (e) => { if (e.target === lb || e.target.classList.contains('lb-close')) lb.style.display = 'none'; };
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') lb.style.display = 'none'; });
  }
  const ann = d.url + '?annotated=1';
  lb.innerHTML = `<div class="lb-box"><div class="lb-bar"><b>${esc(d.id)}</b>` +
    `<label class="chk"><input type="checkbox" id="lb-ann"> geannoteerd</label><button class="lb-close">Sluiten (Esc)</button></div>` +
    `<div class="lb-img"><img src="${d.url}"><div class="lb-mark" style="left:${d.u / d.w * 100}%;top:${d.v / d.h * 100}%"></div></div></div>`;
  lb.style.display = 'flex';
  $('#lb-ann').onchange = (e) => { lb.querySelector('img').src = e.target.checked ? ann : d.url; };
}

function shotHTML(sh, annotated) {
  const place = encodeURIComponent(state.sv.place);
  const id = encodeURIComponent(sh.id);
  const ann = annotated && sh.annotated !== false;
  return `<div class="shot-box" data-id="${esc(sh.id)}"><label class="chk"><input type="checkbox" class="ann" ${ann ? 'checked' : ''}> geannoteerd</label>` +
    `<img class="shot" src="/api/sv/shot/${place}/${id}.png?annotated=${ann ? 1 : 0}" alt="opname ${esc(sh.id)}"></div>`;
}

function wireShot() {
  document.querySelectorAll('.shot-box').forEach((b) => {
    const cb = b.querySelector('.ann'), img = b.querySelector('img');
    cb.onchange = () => {
      img.src = `/api/sv/shot/${encodeURIComponent(state.sv.place)}/${encodeURIComponent(b.dataset.id)}.png?annotated=${cb.checked ? 1 : 0}`;
    };
  });
}

function codeText(c) { return c == null ? '' : (Array.isArray(c) ? c.join('+') : c); }

function editForm(obj) {
  const ov = obj.props.override || {};
  const aps = state.objects.approaches || [];
  const apOpts = ['<option value="">—</option>'].concat(aps.map((a, i) =>
    `<option value="${i}" ${ov.approach === i ? 'selected' : ''}>A${i}: ${esc(a.name || '')}</option>`)).join('');
  const isSign = obj.kind === 'sign';
  return `<h2 style="margin-top:12px">Aanpassen</h2>
  <div class="form">
    <label>x</label><input type="number" step="0.1" id="e-x" value="${obj.at[0].toFixed(2)}">
    <label>y</label><input type="number" step="0.1" id="e-y" value="${obj.at[1].toFixed(2)}">
    <label>richting °</label><input type="number" step="1" id="e-h" value="${(obj.heading ?? 0).toFixed(1)}">
    ${isSign ? `<label>code</label><input type="text" id="e-code" value="${esc(codeText(obj.props.code))}" placeholder="bv. C2 of B6+OB01">` : ''}
    <label>aanrij&shy;richting</label><select id="e-ap">${apOpts}</select>
    <label>notitie</label><input type="text" id="e-note" value="${esc(ov.note || '')}">
  </div>
  <div class="row">
    <button id="e-move">Verplaatsen (g)</button><button id="e-rot">Draaien (r)</button>
  </div>
  <div class="row">
    <button class="primary" id="e-save">Override opslaan</button>
    ${obj.props.override ? '<button id="e-reset">Override wissen</button>' : ''}
    <button class="danger" id="e-remove">${obj.props.removed ? 'Terugzetten' : 'Weghalen'}</button>
  </div>
  <div class="muted small">Opslaan schrijft ref/overrides.json; zichtbaar in het model na <b>Herbouwen</b>.</div>`;
}

function wireEdit(obj) {
  const e = state.byId.get(obj.id);
  state.edit = { obj, group: e.group, pos0: e.group.position.clone(), rot0: e.group.rotation.z };
  const syncFromInputs = () => {
    const g = state.edit.group;
    g.position.x = Number($('#e-x').value);
    g.position.y = Number($('#e-y').value);
    g.position.z = groundZ(g.position.x, g.position.y);
    g.rotation.z = -Number($('#e-h').value) * Math.PI / 180;
  };
  ['#e-x', '#e-y', '#e-h'].forEach((s) => $(s).addEventListener('input', syncFromInputs));
  $('#e-ap').onchange = () => {
    const v = $('#e-ap').value;
    if (v === '') return;
    const a = state.objects.approaches[Number(v)];
    const dx = a.right[0] - a.left[0], dy = a.right[1] - a.left[1], n = Math.hypot(dx, dy) || 1;
    const f = [dy / n, -dx / n];                     // (facing the way in's traffic: kunstwerk_placement)
    $('#e-h').value = ((Math.atan2(f[0], f[1]) * 180 / Math.PI + 360) % 360).toFixed(1);
    syncFromInputs();
  };
  $('#e-move').onclick = () => setGizmo('translate');
  $('#e-rot').onclick = () => setGizmo('rotate');
  $('#e-save').onclick = () => saveOverride(obj);
  $('#e-remove').onclick = () => saveOverride(obj, { remove: !obj.props.removed });
  if ($('#e-reset')) $('#e-reset').onclick = async () => {
    await postJSON('/api/overrides/delete', { id: obj.id });
    log(`override gewist: ${obj.id}`);
    await reloadKeepSelection(obj.id);
  };
}

function setGizmo(mode) {
  if (!state.edit) return;
  gizmo.attach(state.edit.group);
  gizmo.setMode(mode);
  gizmo.showX = mode === 'translate';
  gizmo.showY = mode === 'translate';
  gizmo.showZ = mode === 'rotate';
}

function onGizmoChange() {
  if (!state.edit) return;
  const g = state.edit.group;
  g.position.z = groundZ(g.position.x, g.position.y);
  g.rotation.x = 0;
  g.rotation.y = 0;
  $('#e-x').value = g.position.x.toFixed(2);
  $('#e-y').value = g.position.y.toFixed(2);
  $('#e-h').value = headingOfRot(g.rotation.z).toFixed(1);
}

function restoreEdited() {
  const { group, pos0, rot0 } = state.edit;
  group.position.copy(pos0);
  group.rotation.set(0, 0, rot0);
}

async function saveOverride(obj, extra = {}) {
  const ov = { id: obj.id };
  const base = obj.props.base_at || obj.at;
  const bf = obj.props.base_facing || obj.facing || [0, 1];
  const x = Number($('#e-x').value), y = Number($('#e-y').value), h = Number($('#e-h').value);
  if (Math.hypot(x - base[0], y - base[1]) > 0.01) ov.at = [x, y];
  const bh = (Math.atan2(bf[0], bf[1]) * 180 / Math.PI + 360) % 360;
  const dh = Math.abs(((h - bh) % 360 + 540) % 360 - 180);
  const ap = $('#e-ap').value;
  if (ap !== '') ov.approach = Number(ap);
  if (dh > 0.1 && ap === '') ov.heading = h;
  else if (dh > 0.1 && ap !== '') {
    const a = state.objects.approaches[Number(ap)];
    const dx = a.right[0] - a.left[0], dy = a.right[1] - a.left[1];
    const ah = (Math.atan2(dy, -dx) * 180 / Math.PI + 360) % 360;
    if (Math.abs(((h - ah) % 360 + 540) % 360 - 180) > 0.1) ov.heading = h;   // (a heading other than the approach's)
  }
  if ($('#e-code')) {
    const c = $('#e-code').value.trim();
    const now = c.includes('+') ? c.split('+').map((s) => s.trim()).filter(Boolean) : c;
    if (c && (c !== codeText(obj.props.code) || (obj.props.override || {}).code)) ov.code = now;
  }
  const note = $('#e-note').value.trim();
  if (note) ov.note = note;
  if (extra.remove != null) ov.remove = extra.remove;
  else if (obj.props.removed) ov.remove = true;
  try {
    await postJSON('/api/overrides', ov);
    log(`override opgeslagen: ${JSON.stringify(ov)}`);
    state.edit = null;
    gizmo.detach();
    await reloadKeepSelection(obj.id);
  } catch (e) {
    log('opslaan mislukt: ' + e.message);
  }
}

async function reloadKeepSelection(id) {
  state.edit = null;
  gizmo.detach();
  await loadObjects();
  select(id);
}

// ---- left panel ------------------------------------------------------------------------------------------------------
function buildLayerList() {
  const box = $('#layers');
  box.innerHTML = '';
  for (const L of LAYERS) {
    const row = document.createElement('label');
    row.className = 'layer';
    row.dataset.key = L.key;
    row.innerHTML = `<input type="checkbox" ${L.on ? 'checked' : ''}><span class="sw" style="background:${L.color}"></span>` +
      `<span>${esc(L.name)}</span><span class="n"></span>`;
    row.querySelector('input').onchange = (e) => {
      groups[L.key].visible = e.target.checked;
      if (L.key === 'osm_ways' && e.target.checked && !groups.osm_ways.children.length) loadWays();
      refreshLabels();
    };
    box.appendChild(row);
  }
  $('#labels').onchange = (e) => { state.showLabels = e.target.checked; refreshLabels(); };
  $('#arrows').onchange = (e) => { state.showArrows = e.target.checked; scene.traverse((o) => { if (o.userData.isArrow) o.visible = state.showArrows; }); };
  $('#legend').innerHTML = Object.values(VERDICT).map(([c, n]) => `<span style="--c:${c}">${esc(n)}</span>`).join('');
  $('#search').oninput = () => {
    const q = $('#search').value.trim().toLowerCase();
    const res = $('#search-results');
    res.innerHTML = '';
    if (q.length < 2) return;
    let n = 0;
    for (const [id, { obj }] of state.byId) {
      if (!(id.toLowerCase().includes(q) || String(obj.label).toLowerCase().includes(q))) continue;
      const d = document.createElement('div');
      d.textContent = `${obj.label} — ${id}`;
      d.onclick = () => { select(id); focusOn(id); };
      res.appendChild(d);
      if (++n >= 30) break;
    }
  };
}

// ---- rebuild & game --------------------------------------------------------------------------------------------------
async function rebuild() {
  const btn = $('#btn-rebuild');
  btn.disabled = true;
  $('#footer')?.classList.remove('collapsed');
  try {
    const r = await (await postJSON('/api/rebuild', { placement: $('#rb-placement').checked })).json();
    log(`== herbouwen gestart (${r.steps.join(' → ')})`);
    $('#log-status').textContent = 'bezig…';
    const es = new EventSource(`/api/rebuild/${r.job}/events`);
    es.onmessage = (ev) => log(JSON.parse(ev.data).line);
    es.addEventListener('done', async (ev) => {
      es.close();
      const ok = JSON.parse(ev.data).ok;
      $('#log-status').textContent = ok ? 'klaar ✓' : 'mislukt ✗';
      btn.disabled = false;
      if (ok) {
        await Promise.all([loadMesh(), loadObjects()]);
        if (state.selected) select(state.selected);
        log('== model en objecten opnieuw geladen');
      }
    });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) btn.disabled = false; };
  } catch (e) {
    log('herbouwen: ' + e.message);
    btn.disabled = false;
  }
}

async function gameStatus() {
  try {
    const s = await (await api('/api/game/status')).json();
    const el = $('#game-status');
    el.textContent = s.reachable ? (s.earth ? 'game: verbonden' : 'game: geen Earth-level') : 'game: niet bereikbaar';
    el.className = 'pill ' + (s.reachable && s.earth ? 'ok' : 'bad');
    el.title = s.reachable ? s.reload_hint : (s.error || '');
  } catch (e) { /* the dashboard's own server gone */ }
}

async function push() {
  const r = await fetch('/api/game/push', { method: 'POST' });
  const d = await r.json();
  log(d.ok ? `game: ${d.result}` : `game: ${d.error}`);
  if (d.ok) log('   (earthd leest de nieuwe export alleen met BNE_KUNSTWERK_RELOAD=1, anders na een herstart)');
}

// ---- start -----------------------------------------------------------------------------------------------------------
async function main() {
  buildLayerList();
  $('#btn-rebuild').onclick = rebuild;
  $('#btn-push').onclick = push;
  $('#log-toggle').onclick = () => document.querySelector('footer').classList.toggle('collapsed');
  state.info = await (await api('/api/info')).json();
  $('#kw-name').textContent = '· ' + state.info.name;
  $('#frame-note').innerHTML = `Lokaal frame: RD − (${state.info.origin_rd.join(', ')}); x oost, y noord, z NAP (m). ` +
    `Richting: kompasgraden, 0 = noord.<br>Muis: links draaien, rechts schuiven, wiel zoomen. g/r: verplaatsen/draaien, f: inzoomen, Esc: los.`;
  const sel = $('#sv-place');
  const places = state.info.sv_places.length ? state.info.sv_places : [state.info.place];
  sel.innerHTML = places.map((p) => `<option ${p === state.info.place ? 'selected' : ''}>${esc(p)}</option>`).join('');
  sel.onchange = () => loadSV(sel.value);
  await loadGround();
  await Promise.all([loadMesh(), loadObjects()]);
  await loadSV(sel.value);
  $('#loading').remove();
  gameStatus();
  setInterval(gameStatus, 20000);
}
main().catch((e) => { $('#loading').textContent = 'Fout: ' + e.message; log(String(e.stack || e)); });
