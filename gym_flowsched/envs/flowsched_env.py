import sys
import numpy as np
from gym import Env, spaces
from gym.envs.toy_text import discrete

class FlowSchedEnv(discrete.DiscreteEnv):
    """
    Description: 
    There is single link network in which flows arrive at different timeslots. When each episode starts, flows arrive one by one, the bandwidth capacity changes, and the agent chooses a protocol for all flows on the link. 

    Initialization:
    There are a total of 20 levels of bandwidth capacity on each link, 3 protocol choices, and 10 flows coming one by one per round. 

    Observations:
    There are 20 states on each link since there are 20 levels of bandwidth capacity on the link. 

    Actions:
    There are 3 actions on the link:
    TCP Cubic,
    TCP Reno,
    TCP Vegas

    Rewards:
    Total achieved rate in the current round

    Probability transition matrix:
    P[s][a]= [(probability, nextstate), ...]

    """




    def __init__(self):
        self.nS = 20
        self.nA = 3
        self.isd = [1/self.nS for x in range(self.nS)]
        self.nF = 10 
        self.rm_size = []
        self.flow_time_link = 0
        self.cum_flowtime = 0

        self.lastaction = None
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed(9)
        self.s = discrete.categorical_sample(self.isd, self.np_random)

        wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
        for j in range(self.nA):
            if j == 1:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
            if j == 2:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]

        self.bandwidth_cap = [i+1 for i in range(self.nS)]
        self.rate = np.matmul(wt,np.diag(self.bandwidth_cap)) # dimension: nA x nS

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                for next_s in range(self.nS):
                    self.P[s][a].append((1/self.nS, next_s))

        self.num_flows = 0

        self.lastaction = None
        #discrete.DiscreteEnv.__init__(self, self.nS, self.nA, P, self.isd)

    def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        self.rm_size = []
        self.flow_time_link = 0
        self.num_flows = 0

        self.seed(7)
        #wt = [[np.random.random() for i in range(self.nS)] for j in range(self.nA)]
        wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
        for j in range(self.nA):
            if j == 1:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
            if j == 2:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]
    
        self.rate = np.matmul(wt,np.diag(self.bandwidth_cap))
        return self.s


    def render(self, mode='human'):
        return self.cum_flowtime
        #print('(Render) rm_size: {}'.format(self.rm_size))

    def _get_flow_time(self, rm_size, flow_time_link, bandwidth_cap, rate):
        """
        rm_size: a vector with a dynamic dimention ranging from 0 to nF
        flow_time_link: a constant
        rate = rate[s][a], a constant
        """
        if rm_size != []:
            i_shortest_flow = np.argmin(rm_size)
        
        time_out = 0
        while time_out < 1 and rm_size != []:
            rate_per_flow = rate / np.size(rm_size) # Needs modification for multiple links 
            if min(rm_size) > rate_per_flow * (1-time_out):
                flow_time_link += (1-time_out) * np.size(rm_size)
                rm_size = [x - rate_per_flow * (1-time_out) for x in rm_size]
                time_out = 1
            else:
                # The following two lines need modification if rate_per_flow is different over flows
                time_shortest_flow = min(rm_size) / rate_per_flow
                flow_time_link += time_shortest_flow * np.size(rm_size)
                rm_size = [x - min(rm_size) for x in rm_size]
                rm_size = [x for x in rm_size if x > 0]
                time_out += time_shortest_flow

        return rm_size, flow_time_link

    def step(self, a):
        if self.num_flows < self.nF:
            self.newflow_size = self.nS # Need to read from a list of flow sizes
            self.rm_size.append(self.newflow_size)
            self.num_flows += 1

        transitions = self.P[self.s][a]
        reward = self.rate[a][self.s]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, newstate = transitions[i]
        self.s = newstate

        wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
        for j in range(self.nA):
            if j == 1:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
            if j == 2:
                wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]

        self.rate = np.matmul(wt,np.diag(self.bandwidth_cap))

        self.rm_size, self.flow_time_link = self._get_flow_time(self.rm_size, self.flow_time_link, self.bandwidth_cap, self.rate[a][self.s])

        if self.rm_size == [] and self.num_flows >= self.nF:
            done = True
            print('Final flow time:{}'.format(self.flow_time_link))
            self.cum_flowtime += self.flow_time_link
            print('Cumulative flow time:{}'.format(self.cum_flowtime))
        else:
            done = False
        return (newstate, reward, done, {"prob": p})

