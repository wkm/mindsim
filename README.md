# MindSim

2-wheeler robot simulation in MuJoCo for training simple neural networks.

## Usage

**This project uses `uv`:**

```bash
uv run python visualize.py
```

See [CLAUDE.md](CLAUDE.md) for complete documentation.

## Future Work

- Try better models (larger networks, different architectures)
- Once the robot consistently learns to navigate to a static target:
  1. Randomize the target location ✓ (curriculum learning, stage 1)
  2. Randomize starting location
  3. Move the target during episodes ✓ (curriculum stage 2)
  4. Add visual distractors ✓ (curriculum stage 3)
  5. Add physical obstacles (walls, ramps)
  6. Multi-target sequences (visit targets in order)
