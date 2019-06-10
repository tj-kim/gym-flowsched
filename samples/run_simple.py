#!/usr/bin/env python

import gym
import gym_flowsched

env = gym.make('FlowSched-v0')
for i_episode in range(2):
    #print('########################################')
    obs = env.reset()
    print('The new epsilode starts')
    env.render()
    for t in range(100):
        #print('--------------------')
        #print(obs)
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        #print('--------------------')
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break