import sys
import numpy as np
from gym.envs.toy_text import discrete

class FlowSchedEnv(discrete.DiscreteEnv):
    """
    Description: 
    There is a network with pre-determined topology in which flows arrive at different timeslots. When each episode starts, flows arrive one by one, the bandwidth capacity on each link changes, and the agent chooses a protocol for all flows on each given link. 

    Initialization:
    There are a total of 20 levels of bandwidth capacity on each link, 3 protocol choices, and 10 flows coming one by one per round. 

    Observations:
    There are 20 states on each link since there are 20 levels of bandwidth capacity on each link. 

    Actions:
    There are 3 actions on each link:
    TCP Cubic,
    TCP Reno,
    TCP Vegas

    Rewards:
    There is a reward of -1 on each alive flow on each link for each action, since as long as the flow is not completed on a given link, the completion time of that flow on that link is increased by 1

    Probability transition matrix:
    P[s][a]= [(probability, nextstate, reward, done), ...]
    
    We first focus on a single link network.

    """




    def __init__(self):
        nS = 20
        nA = 3
        isd = [1/nS for x in range(nS)]
        self.nF = 10 
        self.min_last = 1 * np.random.random()
        self.rm_size = []
        self.flow_time_link = 0

        wt = [[np.random.random() for i in range(nS)] for j in range(nA)]
        self.bandwidth_cap = [i+1 for i in range(nS)]
        self.rate = np.matmul(wt,np.diag(self.bandwidth_cap))
        # dimension: nA x nS

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            for a in range(nA):
                for next_s in range(nS):
                    P[s][a].append((1/nS, next_s, self.rate[a][s]))

        self.num_flows = 0
        discrete.DiscreteEnv.__init__(
            self, nS, nA, P, isd)


    def render(self, mode='human'):
        print(self.flow_time_link)

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
            rate_per_flow = rate / self.num_flows # Needs modification for multiple links 
            if min(rm_size) > rate_per_flow * (1-time_out):
                flow_time_link += (1-time_out) * self.num_flows
                rm_size = [x - rate_per_flow * (1-time_out) for x in rm_size]
            else:
                # The following two lines need modification if rate_per_flow is different over flows
                time_shortest_flow = min(rm_size) / rate_per_flow
                time_out += time_shortest_flow
                flow_time_link += time_shortest_flow * self.num_flows
                rm_size = [x - min(rm_size) for x in rm_size]
                for i in range(np.size(rm_size)):
                    if rm_size[i] <= 0:
                        self.num_flows -= 1
                        rm_size.remove(rm_size[i])

        return rm_size, flow_time_link

    def step(self, a):
        if self.num_flows < self.nF:
            self.newflow_size = self.nS * self.min_last
            self.rm_size.append(self.newflow_size)
            self.num_flows += 1

        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        prob, newstate, reward = transitions[i]
        self.s = newstate

        self.rm_size, self.flow_time_link = self._get_flow_time(self.rm_size, self.flow_time_link, self.bandwidth_cap, self.rate[a][self.s])
 

        if self.rm_size == [] and self.num_flows >= self.nF:
            done = True
        else:
            done = False
        #return (newstate, reward, done, self.flow_time_link, prob)
        return (newstate, reward, done, prob)

