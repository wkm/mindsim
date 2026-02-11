# Experiments Log

Tracking experimental branches and their outcomes.

| Branch                           | Hypothesis                                                                                                             | Outcome                                                                       | W&B Runs |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------- |
| `exp/curriculum-target-distance` | Starting with target in front of robot and gradually randomizing position will help the model learn to seek the target | Merged (71ea1d7) - Curriculum learning now integrated into main training loop | -        |
| `exp/ppo-baseline`               | PPO with learned value baseline will reduce gradient variance and enable learning in long episodes where REINFORCE fails | In progress                                                                   | -        |
