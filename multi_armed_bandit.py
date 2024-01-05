import itertools
import math
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
import psutil
import random
        
class UCB_Learner:

    def __init__(self, G, auctions, T = None):
        
        self.ranking = parallel_page_rank(G, psutil.cpu_count(logical=False))
        self.communities = louvain_communities(G)
        n = len(self.communities)
        print("communities: ", len(self.communities))
        
        self.communities_ranking = dict()
        for index, community in enumerate(self.communities):
            self.communities_ranking[index] = sorted(community, key = lambda x: self.ranking[x], reverse = True)
        
        self.auctions_arms = list(auctions)
        nodes = list(G.nodes())
        
        self.__T = T
        if self.__T is None:
            self.__t = 1 # if the horizon is not specified, we assume t = 1 and register timesteps

        self.__last_played_arm = None
        
        # self.__num_auctions = {a: 0 for a in self.auctions_arms}
        # self.__num_num_nodes = {a: 0 for a in range(1, n+1)}
        self.__num = {a: 0 for a in itertools.product(self.auctions_arms, range(1, n+1))}

        # self.__rew_auctions = {a: 0 for a in self.auctions_arms}
        # self.__rew_num_nodes = {a: 0 for a in range(1, n+1)}
        self.__rew = {a: 0 for a in itertools.product(self.auctions_arms, range(1, n+1))}
        
        # self.__ucb_auctions = {a: float("inf") for a in self.auctions_arms}
        # self.__ucb_num_nodes = {a: float("inf") for a in range(1, n+1)}
        self.__ucb = {a: float("inf") for a in itertools.product(self.auctions_arms, range(1, n+1))}
        
    def __choose_arm(self, __ucb):
        # return max(__ucb_auctions, key = __ucb_auctions.get), max(__ucb_num_nodes, key = __ucb_num_nodes.get)
        return max(__ucb, key = __ucb.get)
    
    def play_arm(self):
        a_t = self.__choose_arm(self.__ucb)
        self.__last_played_arm = a_t
        chosen_auction_arm, chosen_num_nodes = a_t
        # self.__num_auctions[chosen_auction_arm] += 1
        # self.__num_num_nodes[chosen_num_nodes] += 1
        self.__num[a_t] += 1
        # make a set with the top chosen_num_nodes nodes with the highest page rank in each community
        seeds = set()
        for index, community in enumerate(self.communities):
            # take random chosen_num_nodes nodes from the community
            # seeds.update(random.sample(community, chosen_num_nodes))
            seeds.update(self.communities_ranking[index][:chosen_num_nodes])
        # seeds.update(set(list(self.ranking.keys())[:chosen_num_nodes]))
        print("seeds: ", seeds)
        print("auction: ", chosen_auction_arm)
        return seeds, chosen_auction_arm
    
    def receive_reward(self, reward):
        
        a_t = self.__last_played_arm

        # self.__rew_auctions[a_t[0]] += reward
        # self.__rew_num_nodes[a_t[1]] += reward
        
        self.__rew[a_t] += reward
        
        if self.__T is not None:
            # self.__ucb_auctions[a_t[0]] = self.__rew_auctions[a_t[0]]/self.__num_auctions[a_t[0]] + math.sqrt(2*math.log(self.__T)/self.__num_auctions[a_t[0]])
            # self.__ucb_num_nodes[a_t[1]] = self.__rew_num_nodes[a_t[1]]/self.__num_num_nodes[a_t[1]] + math.sqrt(2*math.log(self.__T)/self.__num_num_nodes[a_t[1]])
            self.__ucb[a_t] = self.__rew[a_t]/self.__num[a_t] + math.sqrt(2*math.log(self.__T)/self.__num[a_t])
        else:
            # self.__ucb_auctions[a_t[0]] = self.__rew_auctions[a_t[0]] / self.__num_auctions[a_t[0]] + math.sqrt(2 * math.log(self.__t) / self.__num_auctions[a_t[0]])
            # self.__ucb_num_nodes[a_t[1]] = self.__rew_num_nodes[a_t[1]] / self.__num_num_nodes[a_t[1]] + math.sqrt(2 * math.log(self.__t) / self.__num_num_nodes[a_t[1]])
            self.__ucb[a_t] = self.__rew[a_t] / self.__num[a_t] + math.sqrt(2 * math.log(self.__t) / self.__num[a_t])
            self.__t += 1

        return a_t, reward
    
    