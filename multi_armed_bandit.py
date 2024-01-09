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
import os

class Learner:
    
    def __init__(self, G, auctions, T = None, n = 2):
        
        cpu = os.cpu_count()
        print("CPU: ", cpu)
        # self.ranking = parallel_page_rank(G, os.cpu_count())
        # print("PageRank done")
        # self.communities = louvain_communities(G)
        # print("Communities done")
        
        # communities_ranking = []
        # for index, community in enumerate(self.communities):
        #     communities_ranking.append(sorted(community, key = lambda x: self.ranking[x], reverse = True)[0])
        #     if index == n-1:
        #         break
        
        communities_ranking = [random.sample(list(G.nodes()), 1)[0] for i in range(n)]
        self.ranking = communities_ranking
        
        self.auctions_arms = list(auctions)
        self.nodes_arms = []
        for i in range(n):
            self.nodes_arms.extend(list(itertools.combinations(communities_ranking, i+1)))
        print(list(self.nodes_arms))

        self.__last_played_arm = None
        
        self.arms = list(itertools.product(self.auctions_arms, self.nodes_arms))
        
    def get_ranking(self):
        return self.ranking
        
class UCB_Learner(Learner):

    def __init__(self, G, auctions, T = None, n = 2):
        
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

        return a_t, reward
    
class EpsGreedy_Learner(Learner):

    def __init__(self, G, auctions, T = None, n = 2):
        
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
    
class Exp_3_Learner(Learner):

    def __init__(self, G, auctions, T = None, n = 2):
        
        super().__init__(G, auctions, T, n)
        self.__arms_set = self.arms
        self.__T = T
        self.__gamma = 1/(3*self.__T)
        self.__eps = math.sqrt((1-self.__gamma)*math.log(len(self.__arms_set))/(3*len(self.__arms_set)*self.__T))
        #It saves Hedge weights, that are initially 1
        self.__weights = {a:1 for a in self.__arms_set}

    #It use the exponential function of Hedge to update the weights based on the received rewards
    def __Hedge_update_weights(self, rewards):
        for a in self.__arms_set:
            self.__weights[a] = self.__weights[a]*((1-self.__eps)**(1-rewards[a]))

    #Compute the Hedge distribution: each arm is chosen with a probability that is proportional to its weight
    def __Hedge_compute_distr(self):
        w_sum = sum(self.__weights.values())
        prob = list()
        for i in range(len(self.__arms_set)):
            prob.append(self.__weights[self.__arms_set[i]]/w_sum)

        return prob
    
    def check_p(self):
        # p must be a value between 0 and 1
        for i in range(len(self.p)):
            # if p is nan, set it to 1
            if math.isnan(self.p[i]):
                self.p[i] = 1

    def play_arm(self):
        self.p = self.__Hedge_compute_distr()
        r=random.random()
        # We chose a random arm with probability gamma
        if r <= self.__gamma:
            a_t = random.choice(self.__arms_set)
        else: #and an arm according the Hedge distribution otherwise
            self.check_p()
            a_t = random.choices(self.__arms_set, self.p)[0]

        self.__last_played_arm = a_t
        
        chosen_auction_arm, chosen_nodes = a_t
        print(chosen_auction_arm, chosen_nodes)
        return set(chosen_nodes), chosen_auction_arm
    
    def receive_reward(self, reward):
        a_t = self.__last_played_arm
        # We compute the fake rewards
        fake_rewards = dict()
        for i in range(len(self.__arms_set)):
            a = self.__arms_set[i]
            if a != a_t:
                fake_rewards[a] = 1
            else:
                fake_rewards[a] = 1 - (1-reward)/self.p[i]
        # normalize the rewards between 0 and 1
        max_reward = max(fake_rewards.values())
        for a in fake_rewards.keys():
            fake_rewards[a] /= max_reward
        self.__Hedge_update_weights(fake_rewards)

    #utily function only used in Adversarial_Bandit script for didactic purposes
    def get_p(self):
        return self.__Hedge_compute_distr()

#COMPUTE OPTIMAL ARM
def compute_reward(T, a, table_cost):
    cum_reward = 0
    for i in range(0,T):
        cum_reward += table_cost[a][i]
    return cum_reward

def compute_opt_arm(T, arms_set, table_cost):
    opt_reward = -float('inf')
    for a in arms_set:
        temp_opt_reward = compute_reward(T, a, table_cost)
        if temp_opt_reward >= opt_reward:
            opt_reward = temp_opt_reward
            opt_arm = a
    return opt_arm