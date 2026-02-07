# MindSim

2-wheeler robot simulation in MuJoCo for training simple neural networks.

## Usage

**This project uses `uv`:**

```bash
uv run python visualize_with_rerun.py
```

See [CLAUDE.md](CLAUDE.md) for complete documentation.

## TODOs

- Try using rerun with directories and a separate file per episode.
- For each episode we want to record a 1d vector for scalars. This means in wandb we'll see a histogram of something like reward over time. Cute.
- If we can get rerun fixed, randomize the initialization state more
- Belief is that the current NN is not normalizing things enough. Need to map outputs to the velocity range. Need to map reward function to a
  more normalized range as well. Probably need to clip? I dunno
- At this point I think we can try better models
- Once the robot consistently learns to navigate to a static target location, lets: 1) randomize the target location; then 2) randomize starting location; then 3) start to move the target location during the episode, slowly.
- Much fun and profit at this point.
