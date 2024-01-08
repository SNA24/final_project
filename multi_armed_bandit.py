import itertools
import math
import random
from social_network_algorithms.centrality_measures.page_rank import parallel_page_rank
from social_network_algorithms.centrality_measures.HITS import parallel_hits_authority
from social_network_algorithms.centrality_measures.HITS import parallel_hits_hubbiness
from social_network_algorithms.centrality_measures.HITS import parallel_hits_both
from social_network_algorithms.centrality_measures.vote_rank import parallel_vote_rank
from social_network_algorithms.centrality_measures.closeness import parallel_closeness
from social_network_algorithms.centrality_measures.degree import parallel_degree
from social_network_algorithms.centrality_measures.betweenness import parallel_betweenness
from social_network_algorithms.centrality_measures.shapley_closeness import parallel_shapley_closeness
from social_network_algorithms.centrality_measures.shapley_degree import parallel_shapley_degree
from social_network_algorithms.centrality_measures.shapley_threshold import parallel_shapley_threshold
from networkx.algorithms.community import louvain_communities

class Learner:
    
    def __init__(self, G, auctions, T = None, n = 1):
        
        self.ranking = parallel_page_rank(G, 10)
        print("PageRank done")
        self.communities = louvain_communities(G)
        print("Communities done")
        
        communities_ranking = []
        for index, community in enumerate(self.communities):
            communities_ranking.append(sorted(community, key = lambda x: self.ranking[x], reverse = True)[0])
            if index == n-1:
                break
        
        self.auctions_arms = list(auctions)
        print(n)
        self.nodes_arms = []
        for i in range(n):
            self.nodes_arms.extend(list(itertools.combinations(communities_ranking, i+1)))
        print(list(self.nodes_arms))

        self.__last_played_arm = None
        
        self.arms = list(itertools.product(self.auctions_arms, self.nodes_arms))
        
class UCB_Learner(Learner):

    def __init__(self, G, auctions, T = None, n = 1):
        
        super().__init__(G, auctions, T, n)
        
        self.__T = T
        if self.__T is None:
            self.__t = 1 # if the horizon is not specified, we assume t = 1 and register timesteps
        
        self.__num = {a: 0 for a in self.arms} # number of times arm a has been played
        self.__rew = {a: 0 for a in self.arms} # sum of the rewards obtained by playing arm a
        self.__ucb = {a: float("inf") for a in self.arms}
    
    def play_arm(self):
        a_t = max(self.__ucb, key = self.__ucb.get)
        self.__last_played_arm = a_t
        chosen_auction_arm, chosen_nodes = a_t
        self.__num[a_t] += 1
        return set(chosen_nodes), chosen_auction_arm
    
    def receive_reward(self, reward):
        
        a_t = self.__last_played_arm

        self.__rew[a_t] += reward
        
        if self.__T is not None:
            self.__ucb[a_t] = self.__rew[a_t]/self.__num[a_t] + math.sqrt(2*math.log(self.__T)/self.__num[a_t])
        else:
            self.__ucb[a_t] = self.__rew[a_t] / self.__num[a_t] + math.sqrt(2 * math.log(self.__t) / self.__num[a_t])
            self.__t += 1

        print("auction: ", a_t[0], "num_nodes: ", a_t[1], "reward: ", reward)
        return a_t, reward
    
class EpsGreedy_Learner(Learner):

    def __init__(self, G, auctions, T = None, n = 1):
        
        super().__init__(G, auctions, T, n)
        
        self.__arms_set = self.arms
        self.__num = {a: 0 for a in self.arms}
        self.__rew = {a: 0 for a in self.arms}
        self.__avgrew = {a: 0 for a in self.arms}
        self.__t = 0 
    
    def play_arm(self, eps):
        r = random.random()
        if r <= eps: #With probability eps_t
            a_t = random.choice(self.__arms_set) #We choose an arm uniformly at random
        else:
            a_t = max(self.__avgrew, key=self.__avgrew.get) #We choose the arm that has the highest average revenue
        self.__last_played_arm = a_t
        chosen_auction_arm, chosen_nodes = a_t
        self.__num[a_t] += 1
        return set(chosen_nodes), chosen_auction_arm
    
    def receive_reward(self, reward):
        
        a_t = self.__last_played_arm

        self.__num[a_t] += 1
        self.__rew[a_t] += reward
        self.__avgrew[a_t] = self.__rew[a_t]/self.__num[a_t]
        self.__t += 1 

        return a_t, reward