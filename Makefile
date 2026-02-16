.PHONY: tui train quick-sim test smoketest view play wt-new wt-ls wt-rm

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

smoketest:
	uv run mjpython main.py smoketest

view:
	uv run mjpython main.py view

play:
	uv run mjpython main.py play

# --- Worktree management ---

WT_DIR = $(shell cd .. && pwd)/mindsim-$(NAME)

wt-new:
ifdef NAME
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Uncommitted changes. Commit before creating a worktree."; \
		git status --short; \
		exit 1; \
	fi
	git worktree add "$(WT_DIR)" -b "exp/$(NAME)"
	@echo ""
	@echo "Worktree created:"
	@echo "  dir:    $(WT_DIR)"
	@echo "  branch: exp/$(NAME)"
	@echo ""
	cd "$(WT_DIR)" && exec claude
else ifdef DESC
	@scripts/wt-new $(DESC)
else
	@echo "Usage:"
	@echo "  make wt-new NAME=my-experiment          # use explicit name"
	@echo "  make wt-new DESC='implement PPO'        # Claude suggests a name"
endif

wt-ls:
	@git worktree list

wt-rm:
	@test -n "$(NAME)" || (echo "Usage: make wt-rm NAME=my-experiment" && exit 1)
	git worktree remove "$(WT_DIR)"
	@echo "Worktree removed: $(WT_DIR)"
	@echo "Branch exp/$(NAME) still exists. Delete with: git branch -d exp/$(NAME)"
