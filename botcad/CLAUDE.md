# botcad

Parametric CAD for robot design. One skeleton → simulation + printable parts + assembly instructions.

## Architecture

Every module is **Data**, **Transformer**, or **Interpreter**. Data flows one direction: Skeleton → transformers enrich → interpreters read. No exceptions.

- **Data never imports interpreters.** If you're tempted to `import` from `emit/` in a data module, the design is wrong.
- **Interpreters never mutate the data model.** They read `Body.world_pos`, `Body.appearance`, `Body.material` — never write.
- **ShapeScript is the only geometry IR.** No module calls build123d directly except `shapescript/backend_occt.py`. Bracket, servo, component geometry all emit ShapeScript programs.

## Rules that prevent real bugs

**`.moved()`, never `.locate()`.** build123d's `.locate()` mutates and returns `self`. On cached solids this silently corrupts every future caller. Always `.moved()`.

**`SetFuzzyValue` + `SetUseOBB` on OCCT booleans.** Without these, `BRepAlgoAPI_Common`/`BRepAlgoAPI_Cut` can infinite-loop on near-tangent faces.

**Insertion channels are NOT envelopes.** `bracket_insertion_channel()` and `cradle_insertion_channel()` cut a path through the parent body for servo insertion during assembly. They are not enclosures. Bracket inserts along -Z; cradle inserts along +X. Cross-section perpendicular to insertion axis must clear the servo.

**Horn sits on the shaft boss, not the body face.** `horn_base_z = shaft_offset[2] + shaft_boss_height`. Horn thickness minimum 2mm.

**Material and Appearance are data on Body/Component.** Emitters read `body.material.density` for mass, `body.appearance.color` for color. Never hardcode `1200.0` or compute color from `body.shape`.

**Component specs match real parts.** Dimensions from manufacturer datasheets and STEP models. Don't approximate.

## ShapeScript pattern

```python
prog = ShapeScript()
box = prog.box(0.01, 0.02, 0.03, tag="shell")
hole = prog.cylinder(0.002, 0.04)
result = prog.cut(box, hole)
prog.output_ref = result
# Consumer: OcctBackend().execute(prog).shapes[prog.output_ref.id]
```

Emit functions return `ShapeScript`. Convenience wrappers like `bracket_solid_solid()` execute the IR and return a `build123d.Solid`.

## Geometry pipeline: Design / Compose / Place / Cut

1. **Design** — primitives in local frame
2. **Compose** — assemble (bracket wraps servo)
3. **Place** — one transform into parent body frame
4. **Cut** — insertion channel uses **same transform** as Place

## Testing

```sh
make validate          # lint + tests + renders (the full check)
make lint              # ruff check + format
uv run pytest tests/ -v  # tests only
```

Shapescript snapshot baselines: `uv run pytest tests/test_shapescript_snapshots.py --snapshot-update`

Key test files:
- `test_component_clearance.py` — sub-part intersection + insertion channel validation
- `test_shapescript_bracket.py` — bracket IR structure + volume
- `test_rom.py` — range-of-motion via MuJoCo simulation
- `test_clearance.py` — body-to-body distance validation
