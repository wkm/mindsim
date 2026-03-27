.PHONY: tui train quick-sim test smoketest view play renders validate wt-new wt-ls wt-rm setup lint test-viewer viewer-cache web

tui:
	uv run mjpython main.py

.DEFAULT_GOAL := tui

quick-sim:
	uv run mjpython main.py quicksim
	rerun quick_sim.rrd

train:
	@# Ensure working tree is clean
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Uncommitted changes. Commit before training."; \
		git status --short; \
		exit 1; \
	fi
	@echo "Git status: clean"
	@echo "Last commit: $$(git log -1 --oneline)"
	@echo ""
	uv run mjpython main.py train

test:
	uv run pytest tests/ -v
	@echo "Running viewer JS tests..."
	@node viewer/tests/test-section-caps.js

smoketest:
	uv run mjpython main.py smoketest

view:
	uv run mjpython main.py view

play:
	uv run mjpython main.py play

renders:
	uv run mjpython scripts/regen_test_renders.py

renders-components:
	uv run mjpython scripts/regen_test_renders.py --components

renders-bots:
	uv run mjpython scripts/regen_test_renders.py --bots

renders-rom:
	uv run mjpython scripts/regen_test_renders.py --rom

validate: lint test renders
	@echo ""
	@echo "Validation complete. Review render diffs with: git diff --stat"

lint:
	uv run ruff check --fix .
	uv run ruff format .
	uv run python tools/lint_project.py
	pnpm exec biome check --write viewer/
	pnpm exec tsc --noEmit

# Port assignment: hash worktree path for deterministic per-worktree ports.
# Main worktree (master/main) gets the defaults (8081/5173).
# Other worktrees get offset ports to avoid conflicts.
API_PORT := $(shell \
	if [ -d .git ]; then echo 8081; \
	else echo $$(( 8082 + $$(pwd | cksum | cut -d' ' -f1) % 100 )); fi)
VITE_PORT := $(shell \
	if [ -d .git ]; then echo 5173; \
	else echo $$(( 5174 + $$(pwd | cksum | cut -d' ' -f1) % 100 )); fi)

web:
	@echo "API port: $(API_PORT)  Vite port: $(VITE_PORT)"
	API_PORT=$(API_PORT) pnpm exec concurrently --kill-others --names api,vite --prefix-colors blue,green \
		"uv run uvicorn mindsim.server:app --host 0.0.0.0 --port $(API_PORT) --reload --reload-dir botcad --reload-dir mindsim" \
		"while ! curl -sf http://localhost:$(API_PORT)/api/bots >/dev/null 2>&1; do sleep 0.2; done && pnpm exec vite --port $(VITE_PORT)"

test-viewer:
	pnpm exec playwright test --config viewer/tests/playwright.config.mjs

setup:
	git config core.hooksPath .githooks
	pnpm install
	pnpm exec playwright install chromium
	@echo "Setup complete (git hooks, node deps, playwright browser)"

# --- Worktree management ---
# TYPE=infra -> infra/<name> branch (no date prefix)
# Default   -> exp/YYMMDD-<name> branch (date auto-prefixed)

WT_DIR = $(shell cd .. && pwd)/mindsim-$(NAME)
DATE_PREFIX = $(shell date +%y%m%d)

ifeq ($(TYPE),infra)
  BRANCH = infra/$(NAME)
else
  BRANCH = exp/$(DATE_PREFIX)-$(NAME)
endif

wt-new:
ifdef NAME
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Uncommitted changes. Commit before creating a worktree."; \
		git status --short; \
		exit 1; \
	fi
	git worktree add "$(WT_DIR)" -b "$(BRANCH)"
	@echo ""
	@echo "Worktree created:"
	@echo "  dir:    $(WT_DIR)"
	@echo "  branch: $(BRANCH)"
	@echo ""
	cd "$(WT_DIR)" && exec claude
else ifdef DESC
	@scripts/wt-new $(DESC)
else
	@echo "Usage:"
	@echo "  make wt-new NAME=my-experiment              # exp/YYMMDD-my-experiment"
	@echo "  make wt-new NAME=better-tui TYPE=infra      # infra/better-tui"
	@echo "  make wt-new DESC='implement PPO'            # Claude suggests a name"
endif

wt-ls:
	@git worktree list

wt-rm:
	@test -n "$(NAME)" || (echo "Usage: make wt-rm NAME=my-experiment" && exit 1)
	git worktree remove "$(WT_DIR)"
	@echo "Worktree removed: $(WT_DIR)"
	@echo "Note: branch still exists. Delete with: git branch -d <branch-name>"
