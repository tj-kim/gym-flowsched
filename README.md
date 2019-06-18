# gym-flowsched
Environment of scheduling flows using the same protocol vector over network links


To check the environment class FlowSchedEnv(discrete.DiscreteEnv), please go to gym_flowsched/envs/flowsched_env.py

## To-do:
Make weight parameters link-specific 

Think about running indepdent A2C for different links; each A2C uses the reward on the corresponding link as input
