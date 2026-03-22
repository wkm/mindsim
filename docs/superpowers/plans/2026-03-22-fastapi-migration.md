# Plan: Migrate Python HTTP server from stdlib to FastAPI + uvicorn

## Why
The current server uses `http.server.ThreadingHTTPServer` with a hand-rolled
`ViewerHTTPHandler` that matches URL patterns with regex. FastAPI + uvicorn
gives us:
- Typed route parameters with automatic validation
- `--reload` (via watchfiles) for instant dev iteration
- Async-ready (though we start sync)
- Cleaner code — no regex routing, no manual header setting

## Current State
- `ViewerHTTPHandler` in `main.py` (~lines 758-1088): 11 API routes + static
  file serving, all using regex-based dispatch in `do_GET`/`do_POST`.
- Caches: `_stl_cache`, `_catalog_json`, `_bots_json`, `_cad_steps_cache`,
  `_bot_cache`, `_solid_cache` — all module-level with threading locks.
- `_run_web_viewer()` (~line 1091): builds component registry, starts
  `ThreadingHTTPServer`.
- `make web`: runs `uv run mjpython main.py web --port 8081 --no-open`,
  then starts Vite on :5173 which proxies `/api` to :8081.

## Plan

### 1. Add dependencies
```
uv add fastapi "uvicorn[standard]" watchfiles
```

### 2. Create `mindsim/server.py`
New file with a FastAPI app containing all routes. This keeps main.py from
growing further and makes the server module independently importable (needed
for `uvicorn mindsim.server:app --reload`).

Structure:
```python
from fastapi import FastAPI, Response, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()

# Module-level caches (moved from main.py)
# _stl_cache, _catalog_json, _bots_json, etc.

# Startup event: build component registry
@app.on_event("startup")  # or lifespan
def startup(): ...

# Routes
@app.get("/api/bots")
@app.get("/api/components")
@app.get("/api/components/{name}/stl/{part}")
@app.get("/api/components/{name}/fasteners")
@app.post("/api/components/{name}/render-svg")
@app.get("/api/bots/{bot}/body/{body}/cad-steps")
@app.get("/api/bots/{bot}/body/{body}/cad-steps/{idx}/stl")
@app.get("/api/bots/{bot}/body/{body}/cad-steps/{idx}/tool-stl")

# Static files (for non-Vite usage)
app.mount("/", StaticFiles(directory=project_root, html=True))
```

Note: Component names can contain colons (e.g. `horn:STS3215`). FastAPI path
params handle this naturally since colons aren't path separators.

### 3. Move helper functions
Move these from main.py to mindsim/server.py:
- `_build_component_registry`, `_component_to_json`, `_component_layers`
- `_generate_solid`, `_generate_stl_bytes`, `_solid_to_stl_bytes`
- `_layer_color`, `_axis_to_quat`, `_LAYER_COLORS`
- `_load_bot`, `_get_cad_steps`
- `_discover_bots`
- All cache globals

Keep in main.py: `_run_web_viewer` (updated to call uvicorn), CLI parsing,
TUI code.

### 4. Update `_run_web_viewer()` in main.py
```python
def _run_web_viewer(bot_name, port, no_open):
    # Pre-flight (bot manifest generation) stays here
    ...
    import uvicorn
    from mindsim.server import app, init_registry
    init_registry()  # build component registry before serving
    if not no_open:
        webbrowser.open(url)
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 5. Update Makefile `web` target
```makefile
web:
    @echo "Starting Python API server on :8081 with hot-reload..."
    @uv run uvicorn mindsim.server:app --host 0.0.0.0 --port 8081 --reload \
        --reload-dir botcad --reload-dir mindsim & echo $$! > /tmp/mindsim-api.pid
    ...
```

Note: We drop `mjpython` for the dev web server since the viewer API doesn't
need MuJoCo's GL context. `mjpython` is only needed for `view`/`play` modes.

### 6. Ensure `mindsim/` is a package
Create `mindsim/__init__.py` if it doesn't exist.

### 7. Static file serving
- In dev: Vite serves static files, proxies `/api` to uvicorn. No change.
- In prod/standalone (`main.py web`): Mount static files via FastAPI's
  `StaticFiles` at `/` as a fallback after API routes.

### 8. Validation
1. `make lint` — ruff check + format
2. `make validate` — full test suite + renders
3. Manual: `make web` and verify API endpoints respond

## Commit Strategy
1. Commit plan
2. Single commit: add deps + create server.py + update main.py + update Makefile
