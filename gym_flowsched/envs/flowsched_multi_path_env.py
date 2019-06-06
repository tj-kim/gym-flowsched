import sys
import numpy as np 
from gym import Env, spaces
from gym.envs.toy_text import discrete

class FlowSchedMultiPathEnv(discrete.DiscreteEnv):
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
    P[s][a]= [(probability, nextstate), ...]
    
    We first focus on a single link network.

    """
    def __init__(self):
        """
        self.rm_size: nL * nF
        newflow_size_link: nF * 1
        self.s: nL * 1
        a: nL * 1
        transitions: nS tuples, where each tuple is (probability, nextstate)
        wt: nA * nS
        self.rate_link: nA * nS
        self.flow_time_link: nL * nF * 1
        self.bw_cap_link: nL * nS
    
        """
        #  Q: how to address the automatic initilization for all links when done == True on any link (i.e., for any i_link)



        # Single dimension environment parameters
        self.nS = 20
        self.nA = 3
        self.isd = [1/self.nS for x in range(self.nS)]
        self.lastaction = None
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.nF = 10
        self.seed(2)
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                for next_s in range(self.nS):
                    self.P[s][a].append((1/self.nS, next_s))
        

        # Parallel environments: one environment for one link
        self.nL = 6        
        self.nS_vec = [self.nS for x in range(nL)]
        self.nA_vec = [3 for x in range(nL)]
        self.rm_size = np.zeros(nL,nF)
        self.flow_time_link = [[0 for x in range(nF)] for y in range(nL)]
        self.num_flows = 0

        for i_link in range(nL):
            self.s[i_link] = discrete.categorical_sample(self.isd, self.np_random)
            wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
            for j in range(self.nA):
                if j == 1:
                    wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
                if j == 2:
                    wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]

            self.bw_cap_link[i_link] = [i+1 for i in range(self.nS)]
            self.rate_link[i_link] = np.matmul(wt,np.diag(self.bw_cap_link[i_link])) 
            # dimension of self.rate_link[i_link]: nA x nS



        def reset(self):
            self.lastaction = None
            self.rm_size = np.zeros((nL,nF))
            self.flow_time_link = [0 for x in range(nF) for y in range(nL)]
            self.num_flows = 0
            for i_link in range(nL):
                self.s[i_link] = discrete.categorical_sample(self.isd, self.np_random)
                self.seed(i_link)
                wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
                for j in range(self.nA):
                    if j == 1:
                        wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
                    if j == 2:
                        wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]
                
                self.bw_cap_link[i_link] = [i+1 for i in range(self.nS)]
                self.rate_link[i_link] = np.matmul(wt,np.diag(self.bw_cap_link[i_link]))
                # dimension of self.rate_link[i_link]: nA x nS
            return self.s

        def render(self, mode='human'):
            return self.flow_time_link

        def _get_flow_time(self, RmSize, FlowTime, Rate):
            """
            RmSize = rm_size[i_link]: nF * 1
            FlowTime = flow_time_link[i_link]: nF * 1
            Rate = rate[i_link][a][s], a constant
            """
           
            RmSize_pos = [x for x in RmSize if x>0]
            rate_per_flow = Rate / np.size(RmSize_pos) 
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
                    FlowTime += (1-time_out) * alive_flag
                    RmSize = [max(x - rate_per_flow * (1-time_out), 0) for x in RmSize]
                    time_out = 1
                else:
                    # The following two lines need modification if rate_per_flow is different over flows
                    time_shortest_flow = min(RmSize_pos) / rate_per_flow
                    FlowTime += time_shortest_flow * np.size(RmSize_pos)
                    RmSize = [max(x - min(RmSize_pos), 0) for x in RmSize]
                    time_out += time_shortest_flow

                RmSize_pos = [x for x in RmSize if x>0]
                rate_per_flow = Rate / np.size(RmSize_pos)  

            return RmSize, FlowTime

        def step(self, a):
            """
            self.rm_size: nL * nF
            newflow_size_link: nF * 1
            self.s: nL * 1
            self.a: nL * 1
            transitions: nS tuples, where each tuple is (probability, nextstate)
            wt: nA * nS
            self.rate_link: nA * nS * nL
            self.flow_time_link: nL * nF 
            self.bandwidth_cap: nL * nS 

            """
            if self.num_flows < self.nF:
                if np.random.random() > 0.5:
                    newflow_size_link = [1, 1, 0, 1, 0, 1] # first path of the 6-link diamond network
                else:
                    newflow_size_link = [1, 0, 1, 0, 1, 0] # second path of the 6-link diamond network
                self.num_flows += 1
                self.rm_size[...,self.num_flows] += newflow_size_link

            for i_link in range(L):

                transitions = self.P[self.s[i_link]][a[i_link]]
                reward = self.rate_link[i_link][a[i_link]][self.s[i_link]]
                i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
                p, newstate = transitions[i]
                p_vec.append(p)
                self.s[i_link] = newstate

                wt = [ [0.2*(np.random.random()-0.5) + 0.9 for i in range(self.nS)] for j in range(self.nA)]
                for j in range(self.nA):
                    if j == 1:
                        wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.7 for i in range(self.nS)]
                    if j == 2:
                        wt[j][0:self.nS] = [0.2*(np.random.random()-0.5) + 0.5 for i in range(self.nS)]

                self.rate_link[i_link] = np.matmul(wt,np.diag(self.bandwidth_cap[i_link]))

                self.rm_size[i_link], self.flow_time_link[i_link] = self._get_flow_time(self.rm_size[i_link], self.flow_time_link[i_link], self.rate[i_link][a][self.s])

                newstate_vec.append(newstate)
                reward_vec.append(reward)

            if ( self.rm_size == np.zeros((nL, nF)) ).all() and self.num_flows >= self.nF:
                done = True
                print('Flow time on link {} is: {}'.format(0, self.flow_time_link[0]))
            else:
                done = False

            return (newstate_vec, reward_vec, done, {"prob": p_vec})



































