import itertools
import math
        
class UCB_Learner:

    def __init__(self, num_seeds, nodes, auctions, T = None):
        
        nodes_arms = list(itertools.combinations(nodes, num_seeds))
        auctions_arms = list(auctions)
        
        self.__T = T
        if self.__T is None:
            self.__t = 1 # if the horizon is not specified, we assume t = 1 and register timesteps

        self.__last_played_arm = None
        
        self.__num_nodes = {a: 0 for a in nodes_arms}
        self.__num_auctions = {a: 0 for a in auctions_arms}
        self.__rew_nodes = {a: 0 for a in nodes_arms}
        self.__rew_auctions = {a: 0 for a in auctions_arms}
        
        self.__ucb_nodes = {a: float("inf") for a in nodes_arms}
        self.__ucb_auctions = {a: float("inf") for a in auctions_arms}
        
    def __choose_arm(self, __ucb_nodes, __ucb_auctions):
        return max(__ucb_nodes, key = __ucb_nodes.get), max(__ucb_auctions, key = __ucb_auctions.get)
    
    def play_arm(self):
        a_t = self.__choose_arm(self.__ucb_nodes, self.__ucb_auctions)
        self.__last_played_arm = a_t
        chosen_node_arm, chosen_auction_arm = a_t
        self.__num_nodes[chosen_node_arm] += 1
        self.__num_auctions[chosen_auction_arm] += 1
        return a_t
    
    def receive_reward(self, reward):
        a_t = self.__last_played_arm
        self.__rew_nodes[a_t[0]] += reward
        self.__rew_auctions[a_t[1]] += reward
        
        if self.__T is not None:
            self.__ucb_nodes[a_t[0]] = self.__rew_nodes[a_t[0]]/self.__num_nodes[a_t[0]] + math.sqrt(2*math.log(self.__T)/self.__num_nodes[a_t[0]])
            self.__ucb_auctions[a_t[1]] = self.__rew_auctions[a_t[1]]/self.__num_auctions[a_t[1]] + math.sqrt(2*math.log(self.__T)/self.__num_auctions[a_t[1]])
        else:
            self.__ucb_nodes[a_t[0]] = self.__rew_nodes[a_t[0]] / self.__num_nodes[a_t[0]] + math.sqrt(2 * math.log(self.__t) / self.__num_nodes[a_t[0]])
            self.__ucb_auctions[a_t[1]] = self.__rew_auctions[a_t[1]] / self.__num_auctions[a_t[1]] + math.sqrt(2 * math.log(self.__t) / self.__num_auctions[a_t[1]])
            self.__t += 1

        return a_t, reward
    
    