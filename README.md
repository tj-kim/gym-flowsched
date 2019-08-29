# gym-flowsched
Environment of scheduling flows using the same protocol vector over network links


To check the environment class FlowSchedEnv(discrete.DiscreteEnv), please go to gym_flowsched/envs/flowsched_env.py

## To-do:
Make weight parameters link-specific 

Think about running indepdent A2C for different links; each A2C uses the reward on the corresponding link as input

## How to add new Env
- `gym_flowsched/__init__.py`: // register new env, let gym_flowsched know
```
register(
    id='FlowSchedXXX',
    entry_point='gym_flowsched.envs:FlowSchedXXXEnv',
    )
```
- `gym_flowsched/envs/__init__.py`: // map new env name to actual location
```
from gym_flowsched.envs.flowsched_xxx_env import FlowSchedXXXEnv
```
- Create `gym_flowsched/envs/flowsched_xxx_env.py` // actual location
