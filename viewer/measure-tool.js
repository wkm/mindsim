/**
 * MindSim Measurement Tool — CAD-style dimension lines.
 *
 * Click two points on visible geometry to create a dimension line
 * with extension lines, arrowheads, and a distance label in mm.
 * Points snap to nearby mesh edges/vertices for precision.
 *
 * Dimension lines are rendered as an SVG overlay that re-projects
 * on every camera change, giving crisp resolution-independent output.
 */

import * as THREE from 'three';
import { BP } from './presentation.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const SNAP_SCREEN_PX = 12;        // snap radius in screen pixels
const ARROW_SIZE = 6;             // arrowhead size in SVG pixels
/** Convert a numeric hex color to a CSS hex string. */
function hexStr(n) { return '#' + n.toString(16).padStart(6, '0'); }

const DIM_COLOR = hexStr(BP.BLUE1);
const DIM_FONT = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const EXTENSION_GAP = 4;          // gap between geometry and extension line start
const EXTENSION_OVERSHOOT = 6;    // how far extension line extends past dim line
const DIM_OFFSET = 20;            // offset of dim line from geometry (screen px)

// ---------------------------------------------------------------------------
// MeasureTool class
// ---------------------------------------------------------------------------
export class MeasureTool {
  /**
   * @param {THREE.Camera} camera
   * @param {THREE.Scene} scene
   * @param {HTMLElement} container — the canvas container element
   */
  constructor(camera, scene, container) {
    this.camera = camera;
    this.scene = scene;
    this.container = container;
    this.enabled = false;

    this.measurements = [];     // array of { p1: Vector3, p2: Vector3, id: number }
    this._nextId = 0;
    this._firstPoint = null;    // Vector3 or null (state machine)
    this._hoverPoint = null;    // current snap candidate
    this._raycaster = new THREE.Raycaster();
    this._mouse = new THREE.Vector2();

    // Edge vertex cache: mesh uuid → Float32Array of edge vertices
    this._edgeVertexCache = new Map();

    // Create SVG overlay
    this._svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this._svg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:90;overflow:visible';
    this._svg.setAttribute('id', 'measure-overlay');
    container.appendChild(this._svg);

    // Snap indicator (small circle shown on hover)
    this._snapIndicator = document.createElement('div');
    this._snapIndicator.style.cssText = `
      position:absolute; width:8px; height:8px; border-radius:50%;
      border:2px solid ${DIM_COLOR}; background:rgba(14,90,138,0.3);
      pointer-events:none; display:none; z-index:91;
      transform:translate(-50%,-50%);
    `;
    container.appendChild(this._snapIndicator);

    // Rubber-band line (from first point to cursor)
    this._rubberLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    this._rubberLine.setAttribute('stroke', DIM_COLOR);
    this._rubberLine.setAttribute('stroke-width', '1');
    this._rubberLine.setAttribute('stroke-dasharray', '4,3');
    this._rubberLine.style.display = 'none';
    this._svg.appendChild(this._rubberLine);

    // Bind handlers (stored for removal)
    this._onMouseMove = this._handleMouseMove.bind(this);
    this._onClick = this._handleClick.bind(this);
    this._onKeyDown = this._handleKeyDown.bind(this);
  }

  // -----------------------------------------------------------------------
  // Enable / disable
  // -----------------------------------------------------------------------

  enable() {
    if (this.enabled) return;
    this.enabled = true;
    this.container.style.cursor = 'crosshair';
    this.container.addEventListener('pointermove', this._onMouseMove);
    this.container.addEventListener('pointerdown', this._onClick);
    window.addEventListener('keydown', this._onKeyDown);
  }

  disable() {
    if (!this.enabled) return;
    this.enabled = false;
    this._firstPoint = null;
    this._hoverPoint = null;
    this._snapIndicator.style.display = 'none';
    this._rubberLine.style.display = 'none';
    this.container.style.cursor = '';
    this.container.removeEventListener('pointermove', this._onMouseMove);
    this.container.removeEventListener('pointerdown', this._onClick);
    window.removeEventListener('keydown', this._onKeyDown);
  }

  clearAll() {
    this.measurements = [];
    this._firstPoint = null;
    this._rubberLine.style.display = 'none';
    this._edgeVertexCache.clear();
    this.update();
  }

  /** Call on every frame or camera change to re-project dimensions. */
  update() {
    // Clear old dimension SVG elements (keep rubber-band line)
    const toRemove = [];
    for (const child of this._svg.children) {
      if (child !== this._rubberLine) toRemove.push(child);
    }
    toRemove.forEach(el => el.remove());

    const rect = this.container.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    for (const m of this.measurements) {
      this._drawDimension(m, w, h);
    }

    // Update rubber-band line
    if (this._firstPoint && this._hoverPoint) {
      const s1 = this._toScreen(this._firstPoint, w, h);
      const s2 = this._toScreen(this._hoverPoint, w, h);
      this._rubberLine.setAttribute('x1', s1.x);
      this._rubberLine.setAttribute('y1', s1.y);
      this._rubberLine.setAttribute('x2', s2.x);
      this._rubberLine.setAttribute('y2', s2.y);
      // Axis color when constrained
      const color = this._axisLocked
        ? this._axisConstraintColor(this._firstPoint, this._hoverPoint)
        : DIM_COLOR;
      this._rubberLine.setAttribute('stroke', color);
      this._rubberLine.style.display = '';
    }
  }

  // -----------------------------------------------------------------------
  // Event handlers
  // -----------------------------------------------------------------------

  _handleMouseMove(e) {
    const rect = this.container.getBoundingClientRect();
    this._mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    let hit = this._findSnapPoint(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height);

    // Shift = axis-constrain relative to first point
    this._axisLocked = false;
    if (hit && this._firstPoint && e.shiftKey) {
      hit = this._constrainToAxis(this._firstPoint, hit);
      this._axisLocked = true;
    }

    if (hit) {
      this._hoverPoint = hit;
      const screen = this._toScreen(hit, rect.width, rect.height);
      this._snapIndicator.style.left = screen.x + 'px';
      this._snapIndicator.style.top = screen.y + 'px';
      this._snapIndicator.style.display = '';
      // Color hint: blue for normal, axis color when constrained
      const constrained = this._firstPoint && e.shiftKey;
      this._snapIndicator.style.borderColor = constrained ? this._axisConstraintColor(this._firstPoint, hit) : DIM_COLOR;
    } else {
      this._hoverPoint = null;
      this._snapIndicator.style.display = 'none';
    }

    // Update rubber-band
    if (this._firstPoint) {
      this.update();
    }
  }

  _handleClick(e) {
    if (e.button !== 0) return;
    if (!this._hoverPoint) return;

    if (!this._firstPoint) {
      // Place first point
      this._firstPoint = this._hoverPoint.clone();
    } else {
      // Place second point → create measurement
      const p2 = this._hoverPoint.clone();
      if (this._firstPoint.distanceTo(p2) > 1e-6) {
        this.measurements.push({
          p1: this._firstPoint,
          p2: p2,
          id: this._nextId++,
        });
      }
      this._firstPoint = null;
      this._rubberLine.style.display = 'none';
      this.update();
    }

    // Prevent OrbitControls from handling this click
    e.stopPropagation();
  }

  _handleKeyDown(e) {
    if (e.key === 'Escape') {
      this._firstPoint = null;
      this._rubberLine.style.display = 'none';
      this.update();
    }
  }

  // -----------------------------------------------------------------------
  // Axis constraint (Shift key)
  // -----------------------------------------------------------------------

  /**
   * Constrain a point to the dominant axis relative to an origin.
   * Projects onto whichever of X, Y, Z has the largest delta.
   */
  _constrainToAxis(origin, point) {
    const dx = Math.abs(point.x - origin.x);
    const dy = Math.abs(point.y - origin.y);
    const dz = Math.abs(point.z - origin.z);

    const result = origin.clone();
    if (dx >= dy && dx >= dz) {
      result.x = point.x;  // X axis
    } else if (dy >= dx && dy >= dz) {
      result.y = point.y;  // Y axis
    } else {
      result.z = point.z;  // Z axis
    }
    return result;
  }

  /** Return a CSS color for the dominant constraint axis. */
  _axisConstraintColor(origin, point) {
    const dx = Math.abs(point.x - origin.x);
    const dy = Math.abs(point.y - origin.y);
    const dz = Math.abs(point.z - origin.z);

    if (dx >= dy && dx >= dz) return hexStr(BP.RED3);
    if (dy >= dx && dy >= dz) return hexStr(BP.GREEN3);
    return hexStr(BP.BLUE4);
  }

  // -----------------------------------------------------------------------
  // Snapping — find nearest edge vertex to mouse position
  // -----------------------------------------------------------------------

  _findSnapPoint(screenX, screenY, viewW, viewH) {
    // First raycast to find which mesh we're near
    this._raycaster.setFromCamera(this._mouse, this.camera);
    const meshes = [];
    this.scene.traverse(child => {
      if (child.isMesh && child.visible && child.parent?.visible !== false) {
        meshes.push(child);
      }
    });
    const intersects = this._raycaster.intersectObjects(meshes, false);

    if (intersects.length === 0) return null;

    const hit = intersects[0];
    const hitPoint = hit.point;

    // Get edge vertices for the hit mesh
    const edgeVerts = this._getEdgeVertices(hit.object);
    if (!edgeVerts || edgeVerts.length === 0) return hitPoint;

    // Find nearest edge vertex in screen space
    let bestDist = Infinity;
    let bestPoint = null;
    const candidate = new THREE.Vector3();

    for (let i = 0; i < edgeVerts.length; i += 3) {
      candidate.set(edgeVerts[i], edgeVerts[i + 1], edgeVerts[i + 2]);
      // Transform to world space if mesh has a transform
      candidate.applyMatrix4(hit.object.matrixWorld);

      const screen = this._toScreen(candidate, viewW, viewH);
      const dx = screen.x - screenX;
      const dy = screen.y - screenY;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < bestDist) {
        bestDist = dist;
        bestPoint = candidate.clone();
      }
    }

    // Also check nearest point on edges (not just vertices)
    const edgeSnap = this._snapToEdge(hit.object, edgeVerts, screenX, screenY, viewW, viewH);
    if (edgeSnap && edgeSnap.dist < bestDist) {
      bestDist = edgeSnap.dist;
      bestPoint = edgeSnap.point;
    }

    if (bestDist < SNAP_SCREEN_PX && bestPoint) {
      return bestPoint;
    }

    // Fallback: return the raw surface hit point
    return hitPoint;
  }

  _snapToEdge(mesh, edgeVerts, screenX, screenY, viewW, viewH) {
    // Edge vertices come in pairs (each line segment = 2 vertices × 3 components)
    let bestDist = Infinity;
    let bestPoint = null;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();

    for (let i = 0; i < edgeVerts.length; i += 6) {
      a.set(edgeVerts[i], edgeVerts[i + 1], edgeVerts[i + 2]).applyMatrix4(mesh.matrixWorld);
      b.set(edgeVerts[i + 3], edgeVerts[i + 4], edgeVerts[i + 5]).applyMatrix4(mesh.matrixWorld);

      const sa = this._toScreen(a, viewW, viewH);
      const sb = this._toScreen(b, viewW, viewH);

      // Project screen point onto screen-space line segment
      const dx = sb.x - sa.x;
      const dy = sb.y - sa.y;
      const lenSq = dx * dx + dy * dy;
      if (lenSq < 1) continue;

      let t = ((screenX - sa.x) * dx + (screenY - sa.y) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));

      const projX = sa.x + t * dx;
      const projY = sa.y + t * dy;
      const dist = Math.sqrt((screenX - projX) ** 2 + (screenY - projY) ** 2);

      if (dist < bestDist) {
        bestDist = dist;
        // Interpolate in 3D between the world-space edge endpoints
        bestPoint = a.clone().lerp(b, t);
      }
    }

    return bestPoint ? { point: bestPoint, dist: bestDist } : null;
  }

  _getEdgeVertices(mesh) {
    const uuid = mesh.uuid;
    if (this._edgeVertexCache.has(uuid)) {
      return this._edgeVertexCache.get(uuid);
    }

    if (!mesh.geometry) return null;

    const edgesGeom = new THREE.EdgesGeometry(mesh.geometry, 28);
    const posAttr = edgesGeom.getAttribute('position');
    const verts = posAttr ? new Float32Array(posAttr.array) : null;
    edgesGeom.dispose();

    this._edgeVertexCache.set(uuid, verts);
    return verts;
  }

  // -----------------------------------------------------------------------
  // Screen projection
  // -----------------------------------------------------------------------

  _toScreen(point, viewW, viewH) {
    const v = point.clone().project(this.camera);
    return {
      x: (v.x * 0.5 + 0.5) * viewW,
      y: (-v.y * 0.5 + 0.5) * viewH,
    };
  }

  // -----------------------------------------------------------------------
  // SVG dimension drawing
  // -----------------------------------------------------------------------

  _drawDimension(measurement, viewW, viewH) {
    const { p1, p2 } = measurement;
    const s1 = this._toScreen(p1, viewW, viewH);
    const s2 = this._toScreen(p2, viewW, viewH);

    // Distance in mm (geometry is in meters)
    const dist3d = p1.distanceTo(p2) * 1000;
    const label = dist3d < 1 ? dist3d.toFixed(2) + ' mm' : dist3d.toFixed(1) + ' mm';

    // Determine offset direction (perpendicular to the line between points)
    const dx = s2.x - s1.x;
    const dy = s2.y - s1.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 2) return;

    // Perpendicular unit vector (pointing "outward")
    const nx = -dy / len;
    const ny = dx / len;

    // Offset the dimension line away from the geometry
    const off = DIM_OFFSET;
    const d1 = { x: s1.x + nx * off, y: s1.y + ny * off };
    const d2 = { x: s2.x + nx * off, y: s2.y + ny * off };

    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'dimension');

    // Extension lines (from geometry point to dimension line, with gap)
    const overFrac = 1 + EXTENSION_OVERSHOOT / off;
    this._svgLine(g, s1.x + nx * EXTENSION_GAP, s1.y + ny * EXTENSION_GAP,
                      s1.x + nx * off * overFrac, s1.y + ny * off * overFrac, '0.5');
    this._svgLine(g, s2.x + nx * EXTENSION_GAP, s2.y + ny * EXTENSION_GAP,
                      s2.x + nx * off * overFrac, s2.y + ny * off * overFrac, '0.5');

    // Dimension line with arrows
    this._svgLine(g, d1.x, d1.y, d2.x, d2.y, '1');

    // Arrowheads
    this._svgArrow(g, d1.x, d1.y, dx / len, dy / len);
    this._svgArrow(g, d2.x, d2.y, -dx / len, -dy / len);

    // Label (centered on dimension line)
    const mid = { x: (d1.x + d2.x) / 2, y: (d1.y + d2.y) / 2 };
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    // Keep text readable (not upside down)
    const textAngle = (angle > 90 || angle < -90) ? angle + 180 : angle;

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', mid.x);
    text.setAttribute('y', mid.y - 4);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', DIM_COLOR);
    text.setAttribute('style', `font: ${DIM_FONT}`);
    text.setAttribute('transform', `rotate(${textAngle}, ${mid.x}, ${mid.y})`);

    // Background rect for readability
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    text.textContent = label;
    g.appendChild(text);

    // Measure text and add background (after adding to DOM for measurement)
    this._svg.appendChild(g);
    const bbox = text.getBBox();
    bg.setAttribute('x', bbox.x - 3);
    bg.setAttribute('y', bbox.y - 1);
    bg.setAttribute('width', bbox.width + 6);
    bg.setAttribute('height', bbox.height + 2);
    bg.setAttribute('fill', 'rgba(245,248,250,0.85)');
    bg.setAttribute('rx', '2');
    bg.setAttribute('transform', text.getAttribute('transform'));
    g.insertBefore(bg, text);
  }

  _svgLine(parent, x1, y1, x2, y2, strokeWidth = '1') {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x1);
    line.setAttribute('y1', y1);
    line.setAttribute('x2', x2);
    line.setAttribute('y2', y2);
    line.setAttribute('stroke', DIM_COLOR);
    line.setAttribute('stroke-width', strokeWidth);
    parent.appendChild(line);
  }

  _svgArrow(parent, x, y, dx, dy) {
    // Arrowhead pointing in direction (dx, dy)
    const s = ARROW_SIZE;
    const px = -dy;
    const py = dx;
    const tip = `${x},${y}`;
    const left = `${x - dx * s + px * s * 0.4},${y - dy * s + py * s * 0.4}`;
    const right = `${x - dx * s - px * s * 0.4},${y - dy * s - py * s * 0.4}`;
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', `${tip} ${left} ${right}`);
    poly.setAttribute('fill', DIM_COLOR);
    parent.appendChild(poly);
  }
}
