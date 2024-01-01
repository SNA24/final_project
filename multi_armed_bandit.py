import itertools
import math
        
class UCB_Learner:

    def __init__(self, num_seeds, nodes, auctions, T = None):
        nodes_arms = list(itertools.combinations(nodes, num_seeds))
        auctions_arms = list(auctions)
        self.__T = T
        if self.__T is None:
            self.__t = 1 # if the horizon is not specified, we assume t = 1 and register timesteps
            
        combined_arms = list(itertools.product(nodes_arms, auctions_arms))
        self.__last_played_arm = None
        
        self.__num = {a: 0 for a in combined_arms} # to save the number of times a specific arm has been chosen
        self.__rew = {a: 0 for a in combined_arms} # to save the reward of each arm
        
        # save the ucb value of each arm until the current timestep
        # init to infin in order to allow each arm to be chosen at least once
        self.__ucb = {a: float("inf") for a in combined_arms}
        
    def __choose_arm(self, __ucb):
        # choose the arm with the hightest ucb value
        return max(__ucb, key = __ucb.get)
    
    def play_arm(self):
        a_t = self.__choose_arm(self.__ucb)
        self.__last_played_arm = a_t
        chosen_node_arm, chosen_auction_arm = a_t
        self.__num[(chosen_node_arm, chosen_auction_arm)] += 1
        return a_t
    
    def receive_reward(self, reward):
        a_t = self.__last_played_arm
        self.__rew[self.__last_played_arm] += reward
        
        if self.__T is not None: #If the time horizon is known
            self.__ucb[a_t] = self.__rew[a_t]/self.__num[a_t] + math.sqrt(2*math.log(self.__T)/self.__num[a_t])
        else: #If the time horizon is unknown, each time step can be the last. Hence, we are more conservative and use t in place of T
            self.__ucb[a_t] = self.__rew[a_t] / self.__num[a_t] + math.sqrt(2 * math.log(self.__t) / self.__num[a_t])
            self.__t += 1

        return a_t, reward
    
    