# TypeScript Style Guide

MindSim viewer TypeScript conventions. Enforced by **Biome** (lint + format) and **tsc** (type checking).

## Toolchain

| Tool | Purpose | Command |
|------|---------|---------|
| `tsc --noEmit` | Type checking | `pnpm exec tsc --noEmit` |
| `biome check` | Lint + format | `pnpm exec biome check --write viewer/` |
| `make lint` | All checks (Python + TS) | Runs ruff, biome, and tsc |

## Formatting (Biome)

- **2-space indent**, single quotes, semicolons always, trailing commas
- **120-char line width** — wider than Python (88) because TS types add length
- Auto-formatted on `biome check --write`

## Type Annotations

### When to annotate

- **Function parameters**: Always. No implicit `any` on public API boundaries.
- **Return types**: On exported functions and class methods. Omit when obvious from a single return statement.
- **Class properties**: Always declare with a type above the constructor.
- **Local variables**: Let TypeScript infer. Annotate only when the inference is wrong or unclear.

### `any` usage

- `any` is allowed during migration. New code should minimize it.
- Prefer `unknown` + narrowing over `any` for external data.
- Three.js callback objects (traverse, raycast) may use `any` — the library's types don't cover dynamic properties like `bodyID`, `geomGroup`.
- Custom mesh properties (`mesh.bodyID`, `mesh.geomGroup`) use `(mesh as any).prop` until we create a typed wrapper.

### Strict mode roadmap

tsconfig currently has `strict: false`. The plan is to enable strict checks incrementally:
1. `noImplicitAny` — after all `any` placeholders are typed
2. `strictNullChecks` — after DOM access patterns use `?.` and `!`
3. `strict: true` — final target

## Naming

- **Files**: `kebab-case.ts` (e.g., `bot-scene.ts`, `explore-mode.ts`)
- **Classes**: `PascalCase` (e.g., `BotScene`, `ExploreMode`)
- **Interfaces/Types**: `PascalCase` (e.g., `BodyState`, `SectionState`)
- **Functions/methods**: `camelCase`
- **Constants**: `UPPER_SNAKE_CASE` for true constants, `camelCase` for module-level config
- **Private members**: `_prefixed` (e.g., `_savedMaterials`, `_handlePointerMove`)
- **Enum members**: `PascalCase`

## Patterns

### Data model + sync

The viewer uses a **data → sync → render** pattern:
- `BotScene` holds all visual state as plain data (no Three.js imports)
- `sync()` maps data model → Three.js scene graph in one idempotent pass
- Modes mutate `BotScene`, then call `sync()`
- Only `sync()` touches mesh visibility/opacity/emissive

### Imports

- Prefer named imports: `import { Thing } from './module.ts'`
- Three.js: `import * as THREE from 'three'` (namespace import)
- Three.js addons: `import { Foo } from 'three/addons/...'`
- Import order (enforced by Biome): external packages, then local modules

### DOM access

- Use `document.getElementById()` with `as HTMLElement` / `as HTMLInputElement` casts
- Prefer `!` post-fix assertion for elements known to exist in index.html
- Use `?.` for elements that may not exist (dynamic UI)

### Classes vs functions

- **Classes** for stateful objects with lifecycle (modes, controllers)
- **Plain functions** for stateless utilities
- **Interfaces** for data shapes (prefer over `type` aliases for object shapes)

## Biome Rules

Key rules and why:

| Rule | Setting | Why |
|------|---------|-----|
| `noExplicitAny` | off | Migration in progress |
| `noNonNullAssertion` | off | DOM elements from index.html are always present |
| `noForEach` | off | `.forEach()` is fine for side-effect-only iteration |
| `noParameterAssign` | off | Common in physics/animation code |
| `noUnusedVariables` | warn | Catch dead code early |
| `noUnusedImports` | warn | Keep imports clean |

## References

- [Biome documentation](https://biomejs.dev/linter/rules/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [Three.js TypeScript guide](https://threejs.org/docs/#manual/en/introduction/TypeScript)
