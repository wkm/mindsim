/**
 * Semantic Viz — type-specific 3D overlays for focused components.
 *
 * Shows contextual visualizations: ROM arcs for joints, bounding boxes
 * for bodies, FOV cones for cameras, rotation arrows for wheels.
 */

import * as THREE from 'three';
import { clearGroup, createArcGeometry, orientToAxis, createTextSprite } from './utils.js';

export class SemanticViz {
  constructor(ctx) {
    this.ctx = ctx;
    this.group = new THREE.Group();
    this.group.name = 'SemanticOverlays';
    this.active = false;
  }

  show() {
    if (!this.active) {
      this.ctx.scene.add(this.group);
      this.active = true;
    }
  }

  hide() {
    this.clear();
    if (this.active) {
      this.ctx.scene.remove(this.group);
      this.active = false;
    }
  }

  clear() {
    clearGroup(this.group);
  }

  /**
   * Show overlays for a joint node.
   * @param {Object} jointData - joint entry from manifest
   * @param {THREE.Box3} bbox - bounding box of the child body
   */
  showJointOverlay(jointData, bbox) {
    this.clear();
    if (!jointData) return;

    const center = new THREE.Vector3();
    bbox.getCenter(center);

    const axis = new THREE.Vector3(...jointData.axis);
    // Swizzle axis from MuJoCo to Three.js convention
    const swizzled = new THREE.Vector3(axis.x, axis.z, -axis.y).normalize();

    const [lo, hi] = jointData.range;
    const arcRadius = 0.05;

    // ROM arc (range of motion)
    if (Math.abs(hi - lo) > 0.01) {
      const rangeGeo = createArcGeometry(arcRadius, lo, hi, 48);
      const rangeArc = new THREE.Line(
        rangeGeo,
        new THREE.LineBasicMaterial({ color: 0x0A6640, transparent: true, opacity: 0.6, linewidth: 2 })
      );
      orientToAxis(rangeArc, swizzled);
      rangeArc.position.copy(center);
      this.group.add(rangeArc);
    }

    // Axis arrow
    const arrow = new THREE.ArrowHelper(swizzled, center, 0.08, 0x0A6640, 0.018, 0.01);
    this.group.add(arrow);

    // Text labels for servo specs
    const specs = jointData.servo_specs;
    if (specs) {
      const labels = [];
      const rangeDeg = jointData.range_deg;
      if (rangeDeg) labels.push(`ROM: ${rangeDeg[0]}° to ${rangeDeg[1]}°`);
      labels.push(`Torque: ${specs.stall_torque_nm.toFixed(2)} N·m`);
      const rpmApprox = (specs.no_load_speed_rad_s * 60 / (2 * Math.PI)).toFixed(0);
      labels.push(`Speed: ${rpmApprox} RPM`);

      const labelPos = center.clone().add(new THREE.Vector3(0.03, 0.025, 0));
      for (let i = 0; i < labels.length; i++) {
        const sprite = createTextSprite(labels[i], { fontSize: 11, color: '#0A6640' });
        sprite.position.copy(labelPos).y -= i * 0.009;
        sprite.scale.set(0.04, 0.01, 1);
        this.group.add(sprite);
      }
    }
  }

  /**
   * Show overlays for a body node.
   * @param {Object} bodyData - body entry from manifest
   * @param {THREE.Box3} bbox - bounding box
   */
  showBodyOverlay(bodyData, bbox) {
    this.clear();
    if (!bodyData || bbox.isEmpty()) return;

    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bbox.getCenter(center);
    bbox.getSize(size);

    // Wireframe bounding box
    const boxGeo = new THREE.BoxGeometry(size.x, size.y, size.z);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    const wireBox = new THREE.LineSegments(
      boxEdges,
      new THREE.LineBasicMaterial({ color: 0x0E5A8A, transparent: true, opacity: 0.5 })
    );
    wireBox.position.copy(center);
    this.group.add(wireBox);
    boxGeo.dispose();

    // Dimension labels
    const dims = bodyData.dimensions;
    if (dims) {
      const dimLabels = [
        `${(dims[0] * 1000).toFixed(1)} mm`,
        `${(dims[1] * 1000).toFixed(1)} mm`,
        `${(dims[2] * 1000).toFixed(1)} mm`,
      ];
      // X dimension
      const xSprite = createTextSprite(dimLabels[0], { fontSize: 10, color: '#DB3737' });
      xSprite.position.set(center.x, center.y - size.y / 2 - 0.008, center.z);
      xSprite.scale.set(0.027, 0.007, 1);
      this.group.add(xSprite);

      // Y dimension
      const ySprite = createTextSprite(dimLabels[1], { fontSize: 10, color: '#0A6640' });
      ySprite.position.set(center.x + size.x / 2 + 0.008, center.y, center.z);
      ySprite.scale.set(0.027, 0.007, 1);
      this.group.add(ySprite);

      // Z dimension
      const zSprite = createTextSprite(dimLabels[2], { fontSize: 10, color: '#137CBD' });
      zSprite.position.set(center.x, center.y, center.z + size.z / 2 + 0.008);
      zSprite.scale.set(0.027, 0.007, 1);
      this.group.add(zSprite);
    }

    // Mass label
    if (bodyData.mass) {
      const massSprite = createTextSprite(`${(bodyData.mass * 1000).toFixed(1)} g`, { fontSize: 11, color: '#BF8C0A' });
      massSprite.position.set(center.x, center.y + size.y / 2 + 0.008, center.z);
      massSprite.scale.set(0.027, 0.007, 1);
      this.group.add(massSprite);
    }
  }

  /**
   * Show overlays for a camera mount.
   * @param {Object} mountData - mount entry from manifest
   * @param {THREE.Vector3} position - world position
   */
  showCameraOverlay(mountData, position) {
    this.clear();
    if (!mountData) return;

    const fovRad = (mountData.fov_deg || 72) * Math.PI / 180;
    const coneLength = 0.12;
    const coneRadius = Math.tan(fovRad / 2) * coneLength;

    const coneGeo = new THREE.ConeGeometry(coneRadius, coneLength, 16, 1, true);
    coneGeo.translate(0, -coneLength / 2, 0);
    coneGeo.rotateX(Math.PI); // point forward
    const cone = new THREE.Mesh(
      coneGeo,
      new THREE.MeshBasicMaterial({ color: 0x7157D9, transparent: true, opacity: 0.15, side: THREE.DoubleSide })
    );
    cone.position.copy(position);
    this.group.add(cone);

    // Cone wireframe
    const wireGeo = new THREE.EdgesGeometry(coneGeo);
    const wire = new THREE.LineSegments(
      wireGeo,
      new THREE.LineBasicMaterial({ color: 0x7157D9, transparent: true, opacity: 0.4 })
    );
    wire.position.copy(position);
    this.group.add(wire);

    // Label
    if (mountData.resolution) {
      const label = createTextSprite(
        `${mountData.resolution[0]}×${mountData.resolution[1]}  FOV ${mountData.fov_deg}°`,
        { fontSize: 11, color: '#7157D9' }
      );
      label.position.copy(position).y += 0.018;
      label.scale.set(0.047, 0.008, 1);
      this.group.add(label);
    }
  }

  /**
   * Show overlays for a wheel mount.
   * @param {Object} mountData - mount entry from manifest
   * @param {THREE.Vector3} position - world position
   */
  showWheelOverlay(mountData, position) {
    this.clear();
    if (!mountData) return;

    // Rotation direction arrow (circular)
    const arcGeo = createArcGeometry(0.04, 0, Math.PI * 1.5, 24);
    const arc = new THREE.Line(
      arcGeo,
      new THREE.LineBasicMaterial({ color: 0x008075, linewidth: 2 })
    );
    arc.position.copy(position);
    this.group.add(arc);

    // Arrowhead at the end of the arc
    const arrowDir = new THREE.Vector3(0, 1, 0);
    const arrowPos = position.clone().add(new THREE.Vector3(0.04, 0, 0));
    const arrowHelper = new THREE.ArrowHelper(arrowDir, arrowPos, 0.015, 0x008075, 0.008, 0.005);
    this.group.add(arrowHelper);
  }
}
