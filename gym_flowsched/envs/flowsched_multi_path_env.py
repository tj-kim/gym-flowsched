import os,sys
import numpy as np
from gym import Env, spaces
from gym.utils import seeding

DEBUG=os.path.isfile('./DEBUG')

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class FlowSchedMultiPathEnv(Env):
    """
    Description:
    Consider a network with pre-determined topology in which flows arrive at
    different timeslots. In each episode, flows arrive one by one, the bandwidth
    capacity on each link changes, and the agent chooses a protocol
    for all flows on each given link.

    Initialization:
    There are in total 20 levels of bandwidth capacity on each link, 3 protocol
    choices, and 10 flows coming one by one per episode.

    Observations:
    There are 20 states on each link corresponding to 20 levels of bandwidth
    capacity on each link.

    Actions:
    There are 3 actions on each link:
    TCP Cubic --> 0
    TCP Reno  --> 1
    TCP Vegas --> 2

    Rewards:
    Rewards represent transmission rates per flow on each link.

    Probability transition matrix on each link i:
    P[ s[i] ][ a[i] ]= [(probability, newstate, reward, done), ...]
    """
    def __init__(self):
        """
        self.nF[1]
        self.rm_size[self.nL, self.nF]: remaining size
        new_flow_size_link[self.nL]: size of new flows on each link
        self.s[self.nL]: state on each link
        a[self.nL]: action on each link
        transitions: self.nS tuples, where each tuple is (probability, nextstate)
        wt[self.nA, self.nS]: #TODO change wt to weight
        self.rate_link: self.nA * self.nS
        self.flow_time_link: self.nL * self.nF * 1
        self.bw_cap_link[self.nL, self.nS]: bandwidth capacity for each state on each link

        """
        self.seed(0)
        #  Q: how to address the automatic initilization for all links when
        #     done == True on any link (i.e., for any ith link)
        self.nL = 6
        self.nF = 10
        self.nS = 20
        self.nA = 3
        self.isd = [ [1/self.nS for _ in range(self.nS)] for _ in range(self.nL)]
        self.action_space = spaces.Box(low=0,
                                       high=2,
                                       shape=(10,),
                                       dtype=np.int64)
        #self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.asarray([0]*self.nL),
                                            np.asarray([self.nS]*self.nL),
                                            dtype=np.int64)
        # Probability transition matrix is the same on each link
        # TODO: change prob from 1/self.nS to something more realistic/gradual
        # TODO: (reward=1, done=False) for now
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                for newstate in range(self.nS):
                    self.P[s][a].append((1/self.nS, newstate, 1, False))

        # Parallel environments: one environment for one link
        self.nS_vec = [self.nS for _ in range(self.nL)]
        self.nA_vec = [3 for _ in range(self.nL)]

        # code reuse
        self.reset()

    def _get_weight(self):
        wt = [ [0.2*(self.np_random.rand()-0.5) + 0.9 for _ in range(self.nS)] for _ in range(self.nA)]
        wt[1][0:self.nS] = [0.2*(self.np_random.rand()-0.5) + 0.7 for _ in range(self.nS)]
        wt[2][0:self.nS] = [0.2*(self.np_random.rand()-0.5) + 0.5 for _ in range(self.nS)]
        return wt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.lastaction = None
        self.rm_size = np.zeros((self.nL,self.nF))
        self.flow_time_link = np.zeros((self.nL, self.nF))
        self.num_flows = 0
        self.s = np.zeros(self.nL, dtype=np.int)
        self.bw_cap_link = np.zeros((self.nL, self.nS))
        self.rate_link = np.zeros((self.nL, self.nA, self.nS))

        wt = self._get_weight()
        for iL in range(self.nL):
            if DEBUG:
                print(np.shape(self.s))
                print(self.s)
            self.s[iL] = categorical_sample(self.isd[iL], self.np_random)
            self.bw_cap_link[iL] = [x+1 for x in range(self.nS)]
            self.rate_link[iL] = np.matmul(wt,np.diag(self.bw_cap_link[iL]))
            # dimension of self.rate_link[i]: nA x nS
        return self.s

    def render(self, mode='human'):
        return self.flow_time_link

    def _get_flow_time(self, RmSize, FlowTime, Rate):
        """
        RmSize = rm_size[i]: nF * 1
        FlowTime = flow_time_link[i]: nF * 1
        Rate = rate[i][a][s], a constant
        """

        RmSize_pos = [x for x in RmSize if x>0]
        rate_per_flow = Rate / (np.size(RmSize_pos) if np.size(RmSize_pos) > 0 else 1)
        time_out = 0

        while time_out < 1 and  RmSize_pos != []:
            alive_flag = []

            for x in RmSize:
                if x>0:
                    alive_flag.append(1)
                else:
                    alive_flag.append(0)

            if min(RmSize_pos) > rate_per_flow * (1-time_out):
                ## FlowTime += (1-time_out) * np.size(RmSize_pos)
                FlowTime += (1-time_out) * alive_flag[-1]
                RmSize = [max(x - rate_per_flow * (1-time_out), 0) for x in RmSize]
                time_out = 1
            else:
                # The following two lines need modification if rate_per_flow is different over flows
                time_shortest_flow = min(RmSize_pos) / rate_per_flow
                FlowTime += time_shortest_flow * np.size(RmSize_pos)
                RmSize = [max(x - min(RmSize_pos), 0) for x in RmSize]
                time_out += time_shortest_flow

            RmSize_pos = [x for x in RmSize if x>0]
            rate_per_flow = Rate / (np.size(RmSize_pos) if np.size(RmSize_pos) > 0 else 1)

        return RmSize, FlowTime

    def step(self, a):
        """
        self.rm_size: self.nL * self.nF
        newflow_size_link: self.nF * 1
        self.s: self.nL * 1
        self.a: self.nL * 1
        transitions: nS tuples, where each tuple is (probability, nextstate)
        wt: self.nA * self.nS
        self.rate_link: self.nA * self.nS * self.nL
        self.flow_time_link: self.nL * self.nF
        self.bandwidth_cap: self.nL * self.nS

        """
        # process one flow per step (?)
        if self.num_flows < self.nF:
            if self.np_random.rand() > 0.5:
                newflow_size_link = [1, 1, 0, 1, 0, 1] # first path of the 6-link diamond network
            else:
                newflow_size_link = [1, 0, 1, 0, 1, 0] # second path of the 6-link diamond network
            self.rm_size[...,self.num_flows] += newflow_size_link
            self.num_flows += 1

        p_vec, newstate_vec, reward_vec = [], [], []
        wt = self._get_weight()
        # round up actions
        a = list(map(lambda x: int(round(x)), a))
        for iL in range(self.nL):
            if DEBUG:
                print(self.s, '|', a)
            transitions = self.P[ self.s[iL] ][ a[iL] ]
            reward = self.rate_link[iL][ a[iL] ][int(self.s[iL])]
            i_trans = categorical_sample([t[0] for t in transitions], self.np_random)
            p, newstate, _, _ = transitions[i_trans]
            p_vec.append(p)
            self.s[iL] = newstate


            self.rate_link[iL] = np.matmul(wt,np.diag(self.bw_cap_link[iL]))

            self.rm_size[iL], self.flow_time_link[iL] = self._get_flow_time(self.rm_size[iL],
                                                                          self.flow_time_link[iL],
                                                                          self.rate_link[iL][a[iL]][int(self.s[iL])])
            newstate_vec.append(newstate)
            reward_vec.append(reward)

        if ( self.rm_size == np.zeros((self.nL, self.nF)) ).all() and self.num_flows >= self.nF:
            done = True
            print('Flow time on link {} is: {}'.format(0, self.flow_time_link[0]))
        else:
            done = False

        #print(newstate_vec, sum(reward_vec), done, {"prob": p_vec})
        return (newstate_vec, sum(reward_vec), done, {"prob": p_vec})

