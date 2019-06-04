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
        self.a: nL * 1
        transitions: nS tuples, where each tuple is (probability, nextstate)
        wt: nA * nS
        self.rate: nA * nS
        self.flow_time_link: nL * nF * 1
        self.cum_flowtime_link: (just for show) nL * 1 
        self.bandwidth_cap: nS * 1
    
        """
        #  Q: how to address the automatic initilization for all links when done == True on any link (i.e., for any i_link)


        self.nS = 20
        self.nA = 3
        self.isd = [1/self.nS for x in range(self.nS)]
        self.nF = 10
        self.rm_size = np.zeros(nL,nF)
        self.nL = 6

        self.flow_time_link = [0 for x in range(nF)] 
        self.cum_flowtime_link = 0

        self.lastaction = None
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.seed(2)
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

        def reset(self):
            self.s = discrete.categorical_sample(self.isd, self.np_random)
            self.lastaction = None
            self.rm_size = []
            self.flow_time_link = 0

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
            return self.flow_time_link

        def _get_flow_time(self, rm_size, flow_time_link, bandwidth_cap, rate):
            pass

        def step(self, a):
            """
            self.rm_size: nL * nF
            newflow_size_link: nF * 1
            self.s: nL * 1
            self.a: nL * 1
            transitions: nS tuples, where each tuple is (probability, nextstate)
            wt: nA * nS
            self.rate: nA * nS
            self.flow_time_link: nL * nF * 1
            self.bandwidth_cap: nS * 1

            """
            newflow_size_link = 

            for i_link in range(L):
                self.rm_size[i_link] += newflow_size_link[i_link]


                transitions = self.P[self.s[i_link]][a[i_link]]
                reward = self.rate[a[i_link]][self.s[i_link]]
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

                self.rate = np.matmul(wt,np.diag(self.bandwidth_cap))

                self.rm_size[i_link], self.flow_time_link[i_link] = self._get_flow_time(self.rm_size[i_link], self.flow_time_link[i_link], self.bandwidth_cap, self.rate[a][self.s])

                if ( self.rm_size[i_link] == np.zeros(nF) ).all() and anymore_flows == 0:
                    done = True
                    print('Flow time on link {} is: {}'.format(i_link, self.flow_time_link[i_link]))
                else:
                    done = False


                newstate_vec.append(newstate)
                reward_vec.append(reward)
                done_vec.append(done)

            return (newstate_vec, reward_vec, done_vec, {"prob": p_vec})



































