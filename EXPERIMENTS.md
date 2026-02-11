# Experiments Log

Tracking experimental branches and their outcomes.

| Branch                           | Hypothesis                                                                                                             | Outcome                                                                       | W&B Runs |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------- |
| `exp/curriculum-target-distance` | Starting with target in front of robot and gradually randomizing position will help the model learn to seek the target | Merged (71ea1d7) - Curriculum learning now integrated into main training loop | -        |
| `exp/biped-walking`              | A 6-joint biped can learn to walk toward targets using the same PPO + curriculum pipeline, with upright/alive rewards and fall detection | In progress | -        |
