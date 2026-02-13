.PHONY: tui train quick-sim test smoketest view

tui:
	uv run python tui.py

.DEFAULT_GOAL := tui

quick-sim:
	uv run python quick_sim.py
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
	uv run python train.py

test:
	uv run pytest tests/ -v

smoketest:
	uv run python train.py --smoketest

view:
	uv run python view.py bots/simple2wheeler/scene.xml
