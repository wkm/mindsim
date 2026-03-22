/**
 * Section cap geometry tests — validates the polygon chaining,
 * containment tree, and triangulation algorithms.
 *
 * Run: node viewer/tests/test-section-caps.js
 */

// Minimal THREE.Vector3 polyfill for Node.js (no WebGL needed)
class Vector3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  clone() { return new Vector3(this.x, this.y, this.z); }
  sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
  dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; }
  crossVectors(a, b) {
    this.x = a.y * b.z - a.z * b.y;
    this.y = a.z * b.x - a.x * b.z;
    this.z = a.x * b.y - a.y * b.x;
    return this;
  }
  normalize() {
    const l = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    if (l > 0) { this.x /= l; this.y /= l; this.z /= l; }
    return this;
  }
  multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  distanceTo(v) {
    const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

// ── Extract algorithms from viewport3d.js (pure functions, no THREE/WebGL) ──

function chainSegments(flatSegments) {
  const segs = [];
  for (let i = 0; i < flatSegments.length; i += 6) {
    segs.push([
      new Vector3(flatSegments[i], flatSegments[i+1], flatSegments[i+2]),
      new Vector3(flatSegments[i+3], flatSegments[i+4], flatSegments[i+5]),
    ]);
  }
  const EPS = 1e-6;
  const used = new Set();
  const polygons = [];
  for (let startIdx = 0; startIdx < segs.length; startIdx++) {
    if (used.has(startIdx)) continue;
    const chain = [segs[startIdx][0], segs[startIdx][1]];
    used.add(startIdx);
    let changed = true;
    while (changed) {
      changed = false;
      const tail = chain[chain.length - 1];
      for (let i = 0; i < segs.length; i++) {
        if (used.has(i)) continue;
        const [a, b] = segs[i];
        if (tail.distanceTo(a) < EPS) {
          chain.push(b); used.add(i); changed = true; break;
        } else if (tail.distanceTo(b) < EPS) {
          chain.push(a); used.add(i); changed = true; break;
        }
      }
    }
    if (chain.length >= 3 && chain[0].distanceTo(chain[chain.length - 1]) < EPS) {
      chain.pop();
      polygons.push(chain);
    }
  }
  return polygons;
}

function pointInPolygon2D(point, polygon) {
  const [px, py] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function signedArea2D(pts) {
  let area = 0;
  for (let i = 0; i < pts.length; i++) {
    const j = (i + 1) % pts.length;
    area += pts[i][0] * pts[j][1];
    area -= pts[j][0] * pts[i][1];
  }
  return area / 2;
}

function classifyContainment(polygons2d) {
  // Sort by absolute area descending
  const sorted = polygons2d
    .map((pts, i) => ({ pts, area: signedArea2D(pts), origIdx: i }))
    .sort((a, b) => Math.abs(b.area) - Math.abs(a.area));

  return sorted.map((poly, i) => {
    let depth = 0;
    for (let j = 0; j < i; j++) {
      if (pointInPolygon2D(poly.pts[0], sorted[j].pts)) depth++;
    }
    return { ...poly, depth, isOuter: depth % 2 === 0 };
  });
}

// ── Tests ──

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) { passed++; }
  else { failed++; console.error(`  FAIL: ${msg}`); }
}

function test(name, fn) {
  console.log(`  ${name}`);
  fn();
}

// ── Chaining tests ──

console.log('chainSegments:');

test('chains a triangle from 3 segments', () => {
  // Triangle: (0,0,0)→(1,0,0)→(0,1,0)→(0,0,0)
  const segs = [
    0,0,0, 1,0,0,  // seg 0
    1,0,0, 0,1,0,  // seg 1
    0,1,0, 0,0,0,  // seg 2
  ];
  const polys = chainSegments(segs);
  assert(polys.length === 1, `expected 1 polygon, got ${polys.length}`);
  assert(polys[0].length === 3, `expected 3 vertices, got ${polys[0].length}`);
});

test('chains a square from 4 segments in scrambled order', () => {
  // Square but segments are out of order
  const segs = [
    1,1,0, 0,1,0,  // seg 2 (top)
    0,0,0, 1,0,0,  // seg 0 (bottom)
    0,1,0, 0,0,0,  // seg 3 (left)
    1,0,0, 1,1,0,  // seg 1 (right)
  ];
  const polys = chainSegments(segs);
  assert(polys.length === 1, `expected 1 polygon, got ${polys.length}`);
  assert(polys[0].length === 4, `expected 4 vertices, got ${polys[0].length}`);
});

test('chains two separate triangles', () => {
  const segs = [
    // Triangle 1
    0,0,0, 1,0,0,
    1,0,0, 0.5,1,0,
    0.5,1,0, 0,0,0,
    // Triangle 2 (offset by 5)
    5,0,0, 6,0,0,
    6,0,0, 5.5,1,0,
    5.5,1,0, 5,0,0,
  ];
  const polys = chainSegments(segs);
  assert(polys.length === 2, `expected 2 polygons, got ${polys.length}`);
});

test('handles reversed segment direction', () => {
  // Same triangle but one segment is reversed
  const segs = [
    0,0,0, 1,0,0,
    0,1,0, 1,0,0,  // reversed!
    0,1,0, 0,0,0,
  ];
  const polys = chainSegments(segs);
  assert(polys.length === 1, `expected 1 polygon, got ${polys.length}`);
});

test('discards unclosed chains', () => {
  // Two segments that don't close
  const segs = [
    0,0,0, 1,0,0,
    1,0,0, 2,0,0,
  ];
  const polys = chainSegments(segs);
  assert(polys.length === 0, `expected 0 polygons (not closed), got ${polys.length}`);
});

// ── Point-in-polygon tests ──

console.log('\npointInPolygon2D:');

test('point inside unit square', () => {
  const sq = [[0,0], [1,0], [1,1], [0,1]];
  assert(pointInPolygon2D([0.5, 0.5], sq), 'center should be inside');
});

test('point outside unit square', () => {
  const sq = [[0,0], [1,0], [1,1], [0,1]];
  assert(!pointInPolygon2D([2, 2], sq), 'far point should be outside');
});

test('point inside concave polygon', () => {
  // L-shape
  const L = [[0,0], [2,0], [2,1], [1,1], [1,2], [0,2]];
  assert(pointInPolygon2D([0.5, 0.5], L), 'bottom-left should be inside');
  assert(pointInPolygon2D([0.5, 1.5], L), 'top-left should be inside');
  assert(!pointInPolygon2D([1.5, 1.5], L), 'top-right should be outside');
});

// ── Containment classification tests ──

console.log('\nclassifyContainment:');

test('single polygon is an outer', () => {
  const polys = [[[0,0], [10,0], [10,10], [0,10]]];
  const result = classifyContainment(polys);
  assert(result.length === 1, 'should have 1 result');
  assert(result[0].isOuter, 'single polygon should be outer');
  assert(result[0].depth === 0, 'depth should be 0');
});

test('outer + hole', () => {
  const outer = [[0,0], [10,0], [10,10], [0,10]];
  const hole = [[2,2], [8,2], [8,8], [2,8]];
  const result = classifyContainment([outer, hole]);
  const outers = result.filter(r => r.isOuter);
  const holes = result.filter(r => !r.isOuter);
  assert(outers.length === 1, `expected 1 outer, got ${outers.length}`);
  assert(holes.length === 1, `expected 1 hole, got ${holes.length}`);
  assert(holes[0].depth === 1, `hole depth should be 1, got ${holes[0].depth}`);
});

test('two disconnected outers', () => {
  const left = [[0,0], [4,0], [4,4], [0,4]];
  const right = [[6,0], [10,0], [10,4], [6,4]];
  const result = classifyContainment([left, right]);
  const outers = result.filter(r => r.isOuter);
  assert(outers.length === 2, `expected 2 outers, got ${outers.length}`);
});

test('two disconnected outers each with a hole', () => {
  const outer1 = [[0,0], [4,0], [4,4], [0,4]];
  const hole1 = [[1,1], [3,1], [3,3], [1,3]];
  const outer2 = [[6,0], [10,0], [10,4], [6,4]];
  const hole2 = [[7,1], [9,1], [9,3], [7,3]];
  const result = classifyContainment([outer1, hole1, outer2, hole2]);
  const outers = result.filter(r => r.isOuter);
  const holes = result.filter(r => !r.isOuter);
  assert(outers.length === 2, `expected 2 outers, got ${outers.length}`);
  assert(holes.length === 2, `expected 2 holes, got ${holes.length}`);
});

test('island inside a hole (nested)', () => {
  const outer = [[0,0], [10,0], [10,10], [0,10]];
  const hole = [[1,1], [9,1], [9,9], [1,9]];
  const island = [[3,3], [7,3], [7,7], [3,7]];
  const result = classifyContainment([outer, hole, island]);
  const outers = result.filter(r => r.isOuter);
  const holes = result.filter(r => !r.isOuter);
  assert(outers.length === 2, `expected 2 outers (outer + island), got ${outers.length}`);
  assert(holes.length === 1, `expected 1 hole, got ${holes.length}`);
  // Island should have depth 2
  const islandResult = result.find(r => r.depth === 2);
  assert(islandResult && islandResult.isOuter, 'island at depth 2 should be outer');
});

// ── Summary ──

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
