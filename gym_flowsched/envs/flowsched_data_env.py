import os,sys
import numpy as np
from numpy import genfromtxt
from gym import Env, spaces
from gym.utils import seeding

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()

class FlowSchedDataEnv(Env):
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
        wt[self.nA, self.nS]: same for all links #TODO change wt to weight
        self.rate_link[self.nA, self.nS]: total achieved transmission rate each link
        self.flow_time_link[self.nL, self.nF]
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
                                       shape=(self.nL,),
                                       dtype=np.int64)
        #self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.asarray([0]*self.nL),
                                            np.asarray([self.nS]*self.nL),
                                            dtype=np.int64)
        # Probability transition matrix is the same on each link
        state_dist = genfromtxt('data/state_dist.txt')
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                for newstate in range(self.nS):
                    self.P[s][a].append((state_dist[newstate], newstate, 1, False))

        # Parallel environments: one environment for one link
        self.nS_vec = [self.nS for _ in range(self.nL)]
        self.nA_vec = [3 for _ in range(self.nL)]


        self.flowtime_episodes = []
        # code reuse
        self.reset()

    def _get_weight(self):
        wt = [ [0.2*(self.np_random.rand()-0.5) + 0.5 for _ in range(self.nS)] for _ in range(self.nA)]
        reno_wt_pdf, cubic_wt_pdf = genfromtxt('data/reno_wt_pdf.txt'), genfromtxt('data/cubic_wt_pdf.txt')
        reno_wt_supports, cubic_wt_supports = genfromtxt('data/reno_wt_supports.txt'), genfromtxt('data/cubic_wt_supports.txt')

        reno_sample_idx  = categorical_sample(reno_wt_pdf, self.np_random)
        cubic_sample_idx = categorical_sample(cubic_wt_pdf, self.np_random)

        calc_sample     = lambda x, y: np.amin(x) + y*(np.amax(x)-np.amin(x)/len(x))

        reno_sample     = calc_sample(reno_wt_supports, reno_sample_idx)  
        cubic_sample    = calc_sample(cubic_wt_supports, cubic_sample_idx)

        wt[0][0:self.nS] = [reno_sample for _ in range(self.nS)]
        wt[1][0:self.nS] = [cubic_sample for _ in range(self.nS)]
        # wt[2][0:self.nS] = [cubic_sample*0.5 for _ in range(self.nS)]
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
            self.s[iL] = categorical_sample(self.isd[iL], self.np_random)
            self.bw_cap_link[iL] = [x+1 for x in range(self.nS)]
            self.rate_link[iL] = np.matmul(wt,np.diag(self.bw_cap_link[iL]))
            # dimension of self.rate_link[iL]: nA x nS
        return self.s

    def render(self, mode='human'):
        return self.flowtime_episodes

    def _get_flow_time(self, RmSize, FlowTime, Rate):
        """
        RmSize[self.nF] = self.rm_size[iL], remaining sizes of each flows on the iL-th link
        FlowTime[self.nF] = self.flow_time_link[iL], flow-times of each flows on the iL-th link
        Rate (constant) = self.rate_link[iL][a][s], total transmission rate of current flows on iL
        """

        RmSize_pos = [x for x in RmSize if x>0]
        num_alive_flows = np.size(RmSize_pos)
        time_out = 0

        while time_out < 1 and  num_alive_flows>0:
            shortest_flowsize = min(RmSize_pos)
            rate_per_flow = Rate / (num_alive_flows if num_alive_flows > 0 else 1) # same rate for all flows
            time_shortest_flow = shortest_flowsize / rate_per_flow
            alive_flag = []

            for x in RmSize:
                if x>0:
                    alive_flag.append(1) # alive flows
                else:
                    alive_flag.append(0) # finished flows or flows that haven't arrived
            alive_flag = np.array(alive_flag)        

            if time_shortest_flow > 1-time_out: # no more flows finished by this timeslot
                FlowTime += (1-time_out) * alive_flag
                RmSize = [max(x - rate_per_flow * (1-time_out), 0) for x in RmSize]
                time_out = 1 # timeslot ends
            else:
                FlowTime += time_shortest_flow * alive_flag
                RmSize = [max(x - shortest_flowsize, 0) for x in RmSize]
                time_out += time_shortest_flow

            # Update the number and sizes of remaining flows and the transmission rate per flow    
            RmSize_pos = [x for x in RmSize if x>0]
            num_alive_flows = np.size(RmSize_pos)

        return RmSize, FlowTime

    def step(self, a):
        # process one flow per step (?)
        if self.num_flows < self.nF:
            if self.np_random.rand() > 0.5:
                path = np.array([1, 1, 0, 1, 0, 1]) # first path of the 6-link diamond network
            else:
                path = np.array([1, 0, 1, 0, 1, 0]) # second path of the 6-link diamond network
            #self.rm_size[...,self.num_flows] += (8 * self.np_random.rand() + 2) * path # assign a new flow onto its path
            self.rm_size[...,self.num_flows] += 10 * path
            self.num_flows += 1

        p_vec, newstate_vec, reward_vec = [], [], []
        wt = self._get_weight()
        # round up actions
        a = list(map(lambda x: int(round(x)), a))
        for iL in range(self.nL):
            transitions = self.P[ self.s[iL] ][ a[iL] ]
            i_trans = categorical_sample([t[0] for t in transitions], self.np_random)
            p, newstate, _, _ = transitions[i_trans]
            p_vec.append(p)
            self.s[iL] = newstate

            self.rate_link[iL] = np.matmul(wt,np.diag(self.bw_cap_link[iL]))
            reward = self.rate_link[iL][ a[iL] ][int(self.s[iL])]

            self.rm_size[iL], self.flow_time_link[iL] = self._get_flow_time(self.rm_size[iL],
                                                                          self.flow_time_link[iL],
                                                                          self.rate_link[iL][a[iL]][int(self.s[iL])])
            newstate_vec.append(newstate)
            reward_vec.append(reward)

        if ( self.rm_size == np.zeros((self.nL, self.nF)) ).all() and self.num_flows >= self.nF:
            print('Flow time on link {} is: {}'.format(0, self.flow_time_link[0]))
            self.flowtime_episodes = sum(self.flow_time_link.max(0))
            done = True
        else:
            done = False

        #print(newstate_vec, sum(reward_vec), done, {"prob": p_vec})
        return (newstate_vec, min(reward_vec), done, {"prob": p_vec})

