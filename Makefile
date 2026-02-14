.PHONY: tui train quick-sim test smoketest view play

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
