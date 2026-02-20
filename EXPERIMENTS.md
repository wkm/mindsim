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
| `exp/260220-simple-hand` | Parallel-jaw gripper on Cartesian rails can learn reach-grasp-lift with PPO + sensors-only MLPPolicy and touch feedback | In progress | - |
