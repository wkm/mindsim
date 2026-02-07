# MindSim

2-wheeler robot simulation in MuJoCo for training simple neural networks.

## Usage

**This project uses `uv`:**

```bash
uv run python visualize_with_rerun.py
```

See [CLAUDE.md](CLAUDE.md) for complete documentation.

## TODO

- Try using rerun with directories and a separate file per episode.
- If we can get rerun fixed, randomize the initialization state more
- Belief is that the current NN is not normalizing things enough. Need to map outputs to the velocity range. Need to map reward function to a
  more normalized range as well. Probably need to clip? I dunno
