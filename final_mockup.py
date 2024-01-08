import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
from social_network_algorithms.mechanisms.SNCA import snca
from social_network_algorithms.mechanisms.VCG import vcg

from multi_armed_bandit import UCB_Learner, EpsGreedy_Learner
import networkx as nx

import random, math
from joblib import Parallel, delayed
from collections import deque

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
        
        self.__learner = UCB_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        # self.__learner = EpsGreedy_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        
        self.__spam = set()
        self.__cache = dict()

    def __init(self, t):
        if type(self.__learner) == UCB_Learner: 
            arms, auction = self.__learner.play_arm()
        else:
            arms, auction = self.__learner.play_arm(give_eps(self.G, t))
        arms = set(arms)
        auction = self.__auctions[[auction["name"] for auction in self.__auctions].index(auction)]
        self.__find_reachable_nodes(arms)
        return arms, auction
    
    def __find_reachable_nodes(self, S):
        
        self.__spam.clear()
        
        if frozenset(S) in self.__cache:
            self.__spam = self.__cache[frozenset(S)]
            return
        
        def bfs (start_node):
            visited, queue = set(), deque([start_node])
            while queue:
                node = queue.pop()
                if node not in visited:
                    visited.add(node)
                    queue.extendleft(set(self.G[node]).difference(visited))
            return set(visited)
        
        if len(S) == 1:
            for s in S:
                spam = bfs(s)
                if len(self.__spam) == 0 and len(S) > 1:
                    self.__spam.update(spam)
                else:
                    self.__spam = spam.intersection(self.__spam)
        else:
            with Parallel(n_jobs=len(S)) as parallel:
                spam = parallel(delayed(bfs)(s) for s in S)
                for s in spam:
                    if len(self.__spam) == 0 and len(S) > 1:
                        self.__spam.update(s)
                    else:
                        self.__spam = s.intersection(self.__spam)

        if frozenset(S) not in self.__cache:
            self.__cache[frozenset(S)] = self.__spam

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
        
    def build_reports_and_bids(self, seed, t, auction, prob, val):
        
        visited = set()
        queue = deque([seed])
        bids = {}
        reports = {}
        
        while queue:
            p = queue.popleft()
            visited.add(p)
            
            for node in self.G[p]:
                if node not in visited and node not in self.__spam and node != p:
                    bid = self.__invite(t, p, node, auction, prob, val)
                    
                    if isinstance(bid, tuple):
                        bids[node] = bid[0]
                        reports[node] = bid[1].difference(self.__spam).difference(self.__S)
                        queue.append(node)  
                    else:
                        if p in reports and node in reports[p]:
                            reports[p].remove(node)
        
        return reports, bids
        
    def run(self, t, prob, val):
        
        self.__S, auction = self.__init(t)

        bids = {seed: {} for seed in self.__S}
        reports = {seed: {} for seed in self.__S}
        
        revenue = 0
        
        if len(self.__S) == 1:
            for seed in self.__S:
                reports[seed], bids[seed] = self.build_reports_and_bids(seed, t, auction, prob, val)
        else:
            with Parallel(n_jobs=len(self.__S)) as parallel:
                res = parallel(delayed(self.build_reports_and_bids)(seed, t, auction, prob, val) for seed in self.__S)
                for seed, r in zip(self.__S, res):
                    reports[seed].update(r[0])
                    bids[seed].update(r[1])

        def build_seller_net_and_run_auction(seed):
            new_seller_net = set()
            for node in self.G[seed]:
                if node in bids[seed].keys():
                    new_seller_net.add(node)
            allocation, payment = auction["auction"](self.k, new_seller_net, reports[seed], bids[seed])
            return allocation, payment
                
        allocation = {}
        payment = {}
        
        if len(self.__S) == 1:
            allocation, payment = build_seller_net_and_run_auction(self.__S.pop())
        else:
            with Parallel(n_jobs=len(self.__S)) as parallel:
                res = parallel(delayed(build_seller_net_and_run_auction)(seed) for seed in self.__S)
                for a, p in res:
                    allocation.update(a)
                    payment.update(p)
        
        for all, pay in zip(allocation.values(), payment.values()):
            if all:
                revenue += pay
                    
        bids.clear()
        reports.clear()
            
        self.__learner.receive_reward(revenue)
        
        print(f"T: {t}, revenue: {revenue}")
            
        return revenue
        
        
