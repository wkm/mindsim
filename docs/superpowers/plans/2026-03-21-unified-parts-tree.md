# Unified Parts Tree — All Bodies with ShapeScript Access

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bot viewer shows every physical object (fabricated and purchased) in a navigable tree. Clicking any item opens its ShapeScript. The tree matches the Assembly > Body > Feature hierarchy.

**Architecture:** Extend the viewer manifest to include ALL parts (servos, horns, fasteners, wires), restructure the component tree to use the Assembly hierarchy, and add ShapeScript navigation from the tree to the split-pane debugger.

**Tech Stack:** Python (manifest generation), JavaScript (viewer tree + navigation), ShapeScript emitters.

---

## The Problem

Currently the bot viewer tree shows ~10 items (kinematic bodies + mounts). The actual bot has 56 physical objects. Servos, horns, fasteners, and wires are rendered in 3D but invisible in the tree — you can't click them, inspect them, or see their ShapeScript.

## The Target Tree

```
wheeler_arm (assembly)
├── base (body, fabricated) → click: ShapeScript debugger
│   ├── LiPo2S-1000 (body, purchased, mounted) → click: ShapeScript
│   ├── RPi Zero 2W (body, purchased, mounted) → click: ShapeScript
│   ├── STS3215 @ left_wheel (body, purchased) → click: ShapeScript
│   │   ├── horn disc (body, purchased) → click: ShapeScript
│   │   └── 4× M2 fastener (body, purchased)
│   ├── STS3215 @ right_wheel (body, purchased) → click: ShapeScript
│   │   ├── horn disc (body, purchased)
│   │   └── 4× M2 fastener (body, purchased)
│   └── wires: servo_bus, power (collapsible)
├── left_wheel (body, fabricated) → click: ShapeScript
├── right_wheel (body, fabricated) → click: ShapeScript
├── arm (sub-assembly)
│   ├── turntable (body, fabricated) → click: ShapeScript
│   ├── STS3215 @ shoulder_yaw (body, purchased) → click: ShapeScript
│   ├── upper_arm (body, fabricated) → click: ShapeScript
│   ├── STS3215 @ elbow (body, purchased) → click: ShapeScript
│   ├── forearm (body, fabricated) → click: ShapeScript
│   └── hand (body, fabricated) → click: ShapeScript
│       └── OV5647 (body, purchased, mounted) → click: ShapeScript
```

Every item is clickable → navigates to `?cadsteps=component:STS3215` or `?cadsteps=wheeler_arm:base`.

## Task Breakdown

### Task 1: Extend viewer manifest with all parts

**Files:**
- Modify: `botcad/emit/viewer.py`

Add to the manifest:

```python
"parts": [
    {
        "id": "servo_left_wheel",
        "name": "STS3215",
        "kind": "purchased",
        "category": "servo",
        "parent_body": "base",
        "joint": "left_wheel",
        "mesh": "servo_STS3215.stl",
        "pos": [x, y, z],
        "quat": [w, x, y, z],
        "shapescript_type": "servo",  # maps to servo_script()
        "shapescript_component": "STS3215",
    },
    {
        "id": "horn_left_wheel",
        "name": "Horn disc",
        "kind": "purchased",
        "category": "horn",
        "parent_body": "base",
        "joint": "left_wheel",
        "mesh": "horn_left_wheel.stl",
        "shapescript_type": "horn",
        "shapescript_component": "STS3215",
    },
    {
        "id": "fastener_left_wheel_0",
        "name": "M2 SHC screw",
        "kind": "purchased",
        "category": "fastener",
        "parent_body": "base",
        "joint": "left_wheel",
        "mesh": "hardware_M2_shc.stl",
    },
    // ... wires, mounted components, etc.
]
```

Each fabricated body already in `"bodies"` gets `"kind": "fabricated"`.
Each mounted component (battery, camera, pi) gets its own entry in `"parts"`.

### Task 2: Rebuild component tree from Assembly hierarchy

**Files:**
- Modify: `viewer/component-tree.js`
- Modify: `viewer/explore-mode.js`

Build the tree from the manifest's assembly structure + parts list:

1. Root = bot assembly
2. Sub-assemblies = collapsible groups
3. Each body (fabricated) = tree node with icon
4. Under each body: its servo, horn, fasteners, mounts as children
5. Wires grouped under a collapsible "Wires" section

Tree node types with distinct icons:
- Assembly (folder icon)
- Body/fabricated (cube icon)
- Body/purchased (cart/package icon)
- Servo (gear icon)
- Horn (disc icon)
- Fastener (screw icon)
- Wire (cable icon)

### Task 3: ShapeScript navigation from tree

**Files:**
- Modify: `viewer/explore-mode.js`
- Modify: `viewer/component-tree.js`

When you click a tree node:
- **Fabricated body** → navigate to `?cadsteps=wheeler_arm:base`
- **Purchased component** → navigate to `?cadsteps=component:STS3215`
- **Assembly** → focus camera on that assembly's bounding box

Add a "View ShapeScript" button or double-click to navigate. Single click = focus + show properties (current behavior).

### Task 4: Update cad-steps-mode for bot body navigation

**Files:**
- Modify: `viewer/cad-steps-mode.js`

The body switcher dropdown should show ALL bodies in the bot, not just structural ones. When switching bodies, reload the ShapeScript for the new body.

### Task 5: Unify component browser into bot viewer

**Files:**
- Modify: `viewer/viewer.js` (URL routing)

The standalone component browser (`?component=STS3215`) and the bot body viewer (`?cadsteps=wheeler_arm:base`) should feel like the same tool. When viewing a component from within a bot context, show a "Back to bot" link.

---

## Ordering

Task 1 (manifest) → Task 2 (tree) → Task 3 (navigation) → Task 4 (body switcher) → Task 5 (unification)

Sequential — each builds on the previous.
