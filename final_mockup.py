import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
from social_network_algorithms.mechanisms.SNCA import snca
from social_network_algorithms.mechanisms.VCG import vcg

from multi_armed_bandit import UCB_Learner, EpsGreedy_Learner, Exp_3_Learner
import networkx as nx

import random, math
from joblib import Parallel, delayed
from collections import deque
import time

def give_eps(G, t):
    if t == 0:
        return 1  #for the first step we cannot make exploitation, so eps_1 = 1
    return (len(G.nodes())*math.log(t+1)/(t+1))**(1/3)

class SocNetMec:
    
    def __init__(self, G, T, k):
        
        self.G = G
        self.T = T
        self.k = k
        
        self.__auctions = [
            
            {
                "name": "MUDAN",
                "auction": mudan,
                "truthful_bidding": True,
                "truthful_reporting": True,
            },
            {
                "name": "SNCA",
                "auction": snca,
                "truthful_bidding": True,
                "truthful_reporting": True,
            },
            {
                "name": "VCG",
                "auction": vcg,
                "truthful_bidding": True,
                "truthful_reporting": True,
            }
            
        ]
        
        # self.__learner = UCB_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        # self.__learner = EpsGreedy_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        self.__learner = Exp_3_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        
        ranking = self.__learner.get_ranking()
        self.cache_bfs = {n: nx.bfs_tree(self.G, n) for n in ranking}
        
        self.__spam = set()
        

    def __init(self, t):
        if type(self.__learner) == UCB_Learner or type(self.__learner) == Exp_3_Learner:
            arms, auction = self.__learner.play_arm()
        else:
            arms, auction = self.__learner.play_arm(give_eps(self.G, t))
        arms = set(arms)
        for a in self.__auctions:
            if a["name"] == auction:
                break
        return arms, a
    
    def __find_reachable_nodes(self, S):
        
        self.__spam.clear()

        if len(S) < 2:
            return [self.cache_bfs[list(S)[0]]]
                    
        trees = []
        
        for index, s in enumerate(list(S)):
            tree = self.cache_bfs[s]
            if index == 0:
                self.__spam = set(tree.nodes())
            else:
                self.__spam = self.__spam.intersection(tree.nodes())
            trees.append(tree)
            
        return trees

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u, v, t):
            bid_v = val(t, v)
            S_v = set(self.G[v]).difference(self.__S)
            if not auction["truthful_bidding"]:
                bid_v = random.randint(0.5*bid_v, bid_v)
            if not auction["truthful_reporting"]:
                S_v = {w for w in S_v if random.random() <= 0.2}
            return bid_v, S_v
        else:
            return False
        
    def build_reports_and_bids(self, seed, t, auction, prob, val, tree):
        
        bids = {}
        reports = {}
        refused = set()
        
        for edge in tree.edges():
            u, v = edge
            if v not in self.__S and v not in self.__spam and u != v:
                res = self.__invite(t, u, v, auction, prob, val)
                if type(res) == tuple:
                    if v in refused:
                        refused.remove(v)
                    bid_v, S_v = res
                    bids[v] = bid_v
                    reports[v] = S_v.difference(self.__spam).difference(self.__S)
                else:
                    refused.add(v)
                
        for v in refused:
            # remove v from reports values
            for w in reports:
                if v in reports[w]:
                    reports[w].remove(v)
            
        return reports, bids
        
    def run(self, t, prob, val):
        
        start = time.time()
        self.__S, auction = self.__init(t)
        
        bids = {s: {} for s in self.__S}
        reports = {s: {} for s in self.__S}
        
        revenue = 0
        trees = self.__find_reachable_nodes(self.__S)
        end = time.time()
        
        print("spam: ", len(self.__spam))
        print(f"init time: {end-start}")
        
        if len(self.__spam) != len(self.G.nodes()):
            start = time.time()
            for index, seed in enumerate(self.__S):
                reports[seed], bids[seed] = self.build_reports_and_bids(seed, t, auction, prob, val, trees[index])
            end = time.time()
            print(f"build reports and bids time: {end-start}")

        def build_seller_net_and_run_auction(seed):
            new_seller_net = set()
            for node in self.G[seed]:
                if node in bids[seed].keys():
                    new_seller_net.add(node)
            allocation, payment = auction["auction"](self.k, new_seller_net, reports[seed], bids[seed])
            return allocation, payment
                
        allocation = {}
        payment = {}
        
        start = time.time()
        for seed in self.__S:
            
            allocation, payment = build_seller_net_and_run_auction(seed)
            
            for node in allocation:
                revenue += payment[node]
                
            allocation.clear()
            payment.clear()
                
        end = time.time()
        print(f"run auction time: {end-start}")
                    
        bids.clear()
        reports.clear()
            
        self.__learner.receive_reward(revenue)
        print("revenue: ", revenue)
            
        return revenue
        
        
