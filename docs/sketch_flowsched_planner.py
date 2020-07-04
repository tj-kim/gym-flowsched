import os,sys
import numpy as np
from gym import Env, spaces
from gym.utils import seeding

class FlowSchedMultiPathEnv(Env):
	def __init__(self):
		"""
		Initialize action and state spaces;
		actions are protocols, 
		states are discretized levels of bandwidth capacities
		"""
	def _get_weight():
		"""
		Randomly produce some parameters 
		associated to transmission rate achievable of each protocol
		"""

		# Read network data (topology, bandwidth, etc) from files
		# Read newly generated flow data from files
		return wt 

	def reset(self):
		"""
		Reset everything (flows status, network characteristics) once an episode is done, 
		meaning all flows are completed
		"""
		pass

	def render(self, mode='human'):
		"""
		Print the flow completion time of each episode 
		"""

		return self.flowtime_episodes

	def _get_flow_time(self, RmSize, Flowtime, Rate):
		"""
		Calculate the total flow completion time of a given link
		accumulated from the episode beginning utill this timestep,
		given the remaining size of each flow on the link
		and the flow completion time updated in the last timestep 
		"""
		return RmSize, Flowtime

	def step(self, actions):
		"""
		Use protocols (actions) from a RL algorithm to process flows:
		First, get the new realized transmission rate based on _get_weight();
		Second, calculate the returned values based on _get_flow_time()
		Third, randomly transition to the next state based on the probabilities in __init__()
		"""

		# Call _get_weight()
		# Call _get_flow_time()
		return (newstate_vec, min(reward_vec), done, {"prob": p_vec})

def main():
	# Please see run_multi_path.py
	
	i_episode = 0
	While True:
		planner = FlowSchedMultiPathEnv(Env)
		# Please see flowsched_data_env.py

		actions, _, _, _ = model.step(obs)
		# actions are protocol solutions 
		# model is pre-trained and updates actions; 
		# it is an external class, 
		# e.g. https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py

		obs, rew, done, _ = planner.step(actions)
		# get observation, reward, and whether an episode is done

		if done:
			i_episode += 1
			if i_episode >= 10000:
				break
			obs = planner.reset()
