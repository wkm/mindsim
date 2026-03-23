/**
 * Unit tests for BotScene — the pure data model.
 *
 * Run with: npx tsx --test viewer/tests/bot-scene.test.ts
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BotScene } from '../bot-scene.ts';

const BODY_NAMES = ['world', 'base', 'turntable', 'upper_arm', 'forearm', 'hand'];

function createScene(n = 6): BotScene {
  return new BotScene(n, BODY_NAMES.slice(0, n));
}

describe('BotScene constructor', () => {
  it('creates N bodies, all visible and not ghosted', () => {
    const scene = createScene();
    assert.equal(scene.bodies.length, 6);
    for (const body of scene.bodies) {
      assert.equal(body.visible, true);
      assert.equal(body.ghosted, false);
      assert.equal(body.hovered, false);
      assert.equal(body.selected, false);
    }
  });

  it('assigns names correctly', () => {
    const scene = createScene();
    assert.equal(scene.bodies[0].name, 'world');
    assert.equal(scene.bodies[3].name, 'upper_arm');
  });

  it('initializes section state', () => {
    const scene = createScene();
    assert.equal(scene.section.enabled, false);
    assert.equal(scene.section.axis, 'y');
    assert.equal(scene.section.fraction, 0.5);
  });
});

describe('BotScene.isolate', () => {
  it('shows only the target body and body 0', () => {
    const scene = createScene();
    scene.isolate(3);

    // Body 3 is visible
    assert.equal(scene.bodies[3].visible, true);
    // Body 0 is visible (parent group)
    assert.equal(scene.bodies[0].visible, true);
    // Others are hidden
    assert.equal(scene.bodies[1].visible, false);
    assert.equal(scene.bodies[2].visible, false);
    assert.equal(scene.bodies[4].visible, false);
    assert.equal(scene.bodies[5].visible, false);
  });

  it('body 0 stays visible during isolate (regression test)', () => {
    const scene = createScene();
    scene.isolate(3);
    assert.equal(scene.bodies[0].visible, true, 'Body 0 must stay visible to avoid hiding all children');
  });

  it('body 0 meshes are hidden during isolate (opacity 0)', () => {
    const scene = createScene();
    scene.isolate(3);
    assert.equal(scene.bodyOpacity(0), 0, 'Body 0 meshes should be hidden during isolation');
  });

  it('isolated body has full opacity', () => {
    const scene = createScene();
    scene.isolate(3);
    assert.equal(scene.bodyOpacity(3), 1.0);
  });

  it('reports isolation state', () => {
    const scene = createScene();
    assert.equal(scene.isIsolated(), false);
    scene.isolate(3);
    assert.equal(scene.isIsolated(), true);
    assert.ok(scene.isolatedIds().has(3));
  });
});

describe('BotScene.isolateMultiple', () => {
  it('shows only specified bodies', () => {
    const scene = createScene();
    scene.isolateMultiple([2, 4]);

    assert.equal(scene.bodies[2].visible, true);
    assert.equal(scene.bodies[4].visible, true);
    assert.equal(scene.bodies[0].visible, true); // parent
    assert.equal(scene.bodies[1].visible, false);
    assert.equal(scene.bodies[3].visible, false);
    assert.equal(scene.bodies[5].visible, false);
  });
});

describe('BotScene.showAll', () => {
  it('resets everything after isolate', () => {
    const scene = createScene();
    scene.isolate(3);
    scene.showAll();

    for (const body of scene.bodies) {
      assert.equal(body.visible, true);
      assert.equal(body.ghosted, false);
    }
    assert.equal(scene.isIsolated(), false);
  });

  it('resets everything after ghost', () => {
    const scene = createScene();
    scene.ghostAllExcept([2]);
    scene.showAll();

    for (const body of scene.bodies) {
      assert.equal(body.ghosted, false);
      assert.equal(body.visible, true);
    }
  });
});

describe('BotScene.bodyOpacity', () => {
  it('invisible body → 0', () => {
    const scene = createScene();
    scene.setBodyVisible(3, false);
    assert.equal(scene.bodyOpacity(3), 0);
  });

  it('ghosted body → 0.06', () => {
    const scene = createScene();
    scene.ghostAllExcept([2]);
    assert.equal(scene.bodyOpacity(3), 0.06);
    assert.equal(scene.bodyOpacity(0), 0.06);
  });

  it('normal body → 1.0', () => {
    const scene = createScene();
    assert.equal(scene.bodyOpacity(3), 1.0);
  });

  it('kept body during ghost → 1.0', () => {
    const scene = createScene();
    scene.ghostAllExcept([2]);
    assert.equal(scene.bodyOpacity(2), 1.0);
  });
});

describe('BotScene.bodyEmissive', () => {
  it('hovered body → highlight color', () => {
    const scene = createScene();
    scene.setHovered(3);
    assert.equal(scene.bodyEmissive(3), 0x666666);
    assert.equal(scene.bodyEmissive(2), 0);
  });

  it('clearing hover removes highlight', () => {
    const scene = createScene();
    scene.setHovered(3);
    scene.setHovered(null);
    assert.equal(scene.bodyEmissive(3), 0);
  });
});

describe('BotScene.ghostAllExcept', () => {
  it('ghosts all bodies except specified ones', () => {
    const scene = createScene();
    scene.ghostAllExcept([1, 3]);

    assert.equal(scene.bodies[0].ghosted, true);
    assert.equal(scene.bodies[1].ghosted, false);
    assert.equal(scene.bodies[2].ghosted, true);
    assert.equal(scene.bodies[3].ghosted, false);
    assert.equal(scene.bodies[4].ghosted, true);
  });
});

describe('BotScene.visibleBodyIds', () => {
  it('returns all IDs when nothing hidden', () => {
    const scene = createScene();
    assert.deepEqual(scene.visibleBodyIds(), [0, 1, 2, 3, 4, 5]);
  });

  it('reflects isolation', () => {
    const scene = createScene();
    scene.isolate(3);
    const visible = scene.visibleBodyIds();
    assert.ok(visible.includes(0));
    assert.ok(visible.includes(3));
    assert.equal(visible.length, 2);
  });
});

describe('BotScene.setBodyVisible', () => {
  it('hides a specific body', () => {
    const scene = createScene();
    scene.setBodyVisible(2, false);
    assert.equal(scene.bodies[2].visible, false);
    assert.equal(scene.bodies[1].visible, true);
  });
});
