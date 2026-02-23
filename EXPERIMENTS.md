# Experiments Log

Tracking experimental branches and their outcomes.

| Branch | Hypothesis | Outcome | W&B Runs |
| --- | --- | --- | --- |
| `exp/curriculum-target-distance` | Starting with target in front of robot and gradually randomizing position will help the model learn to seek the target | Merged (71ea1d7) - Curriculum learning now integrated into main training loop | - |
| `exp/ppo-baseline` | PPO with learned value baseline will reduce gradient variance and enable learning in long episodes where REINFORCE fails | Merged (a8fa5c4) - PPO algorithm, checkpointing, and training improvements | - |
| `exp/no-curriculum-baseline` | Baseline comparison: training without curriculum to measure curriculum's contribution | Merged (1036f51) | - |
| `exp/wandb-tui-integration` | Unified W&B tracking and TUI overhaul for better experiment management | Merged (72b9d26) - Per-run directories, unified W&B, TUI overhaul | - |
| `exp/biped-walking` | A 6-joint biped can learn to walk toward targets using the same PPO + curriculum pipeline, with upright/alive rewards and fall detection | Superseded by `exp/biped-walker-parity` | - |
| `exp/biped-walker-parity` | Aligning biped physics with the Walker2d paradigm will improve walking behavior | In progress | - |
| `exp/modal-remote` | GCP spot VMs with Docker can provide cost-effective remote training infrastructure | In progress | - |
| `exp/reinforce-v2` | Entropy coefficient annealing + episode length annealing will prevent early std collapse and reduce gradient variance in REINFORCE | In progress | - |
| `exp/duck-biped-stability` | Duck-proportioned biped (low CoM, wide stance, short legs, box feet, position actuators) with quantitative stability measurement framework. Hypothesis: a passively stable body design will be dramatically easier to train for walking. | In progress — standing=1.0, impulse=1.0, mobility=0.1, overall=0.73 | - |
| `exp/duck-waddle-body` | Alternative body morphology for biped locomotion | Abandoned (no unique changes) | - |
| `exp/increase-variance-diversity` | Increasing exploration variance/diversity improves sample efficiency | Abandoned (no unique changes) | - |
| `exp/ppo-architecture-baseline` | PPO architecture comparison baseline | Abandoned (no unique changes) | - |
| `exp/walls-exploration` | Adding walls to the environment encourages more directed exploration | Abandoned (no unique changes) | - |
| `exp/260221-scene-gen` | Procedural scene generation: parametric furniture concepts (table, chair, shelf) composed from MuJoCo primitives, placed at random per episode. Phase 1 = primitives with `@lru_cache`, future phases add STL mesh generation with disk caching. Goal: visual diversity + physical obstacles for richer training. Scale progression: room -> apartment -> house -> village. | In progress | - |
| `exp/260222-biped-vision-curriculum` | Switch childbiped from MLPPolicy (sensor-only) to LSTMPolicy (CNN + sensors + LSTM) to enable vision-based navigation. The 5-stage curriculum already exists (walking → angle variance → moving target → static distractors → moving distractors). Walking stage feeds blank images so LSTM learns gait from 34D proprioceptive sensors alone; navigation stages activate the camera. Eased curriculum thresholds (0.8/0.015) vs strict (1.0/0.01) to encourage faster stage progression. **Run 1** (`child-lstm-0222-2238`, W&B 9ndvkyic): 900+ batches, 58k episodes, 0% curriculum advancement. Root cause: alive_bonus=5.0 was 87% of reward (policy stood still); entropy_coeff=0.005 pushed std up; max_grad_norm=10.0 allowed grad norms of 3000+. **Run 2 param changes**: alive_bonus 5→0.5, entropy 0.005→0.0, grad_norm 10→1.0, epochs 10→4, lr 3e-4→1e-4, smoothness 0→0.01, grace 50→125 steps, init_std 0.3→0.5, walking success no longer requires surviving full episode. | In progress | 9ndvkyic |
