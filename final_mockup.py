import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
from social_network_algorithms.mechanisms.SNCA import snca
from social_network_algorithms.mechanisms.VCG import vcg

import random
from joblib import Parallel, delayed
from collections import deque

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
        
        self.__invited_by_seed = dict() # {node: seed1, ...}
        self.__invited_by_nodes = dict() # {node0: {node1, node2, ...}, ...}

    #MOCK-UP IMPLEMENTATION: It assigns the item to the first k bidders and assigns payment 0 to every node
    def __mock_auction(k, seller_net, reports, bids):
        allocation = dict()
        payment = dict()
        count = 0
        for i in bids.keys():
            if count < k:
                allocation[i] = True
            else:
                allocation[i] = False
            payment[i] = 0

    def __choose_S(self):
        # returns a subsets S of G's nodes according to some criteria
        S = set()
        
        while len(S) < 5:
            S.add(random.choice(list(self.G.nodes())))
        
        return S
    
    def __choose_auction(self):
        return self.__auctions[0]

    def __init(self, t):
        return self.__choose_S(), self.__choose_auction()

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
        
    def __build_reports_and_bids(self, seed, bids, reports, t, auction, prob, val):
        
        visited = set()
        queue = deque([seed])

        while queue:
            
            u = queue.popleft()
            visited.add(u)

            # visit all neighbors of u
            for v in self.G.neighbors(u):
                
                # if v is a seed, then skip it
                if v in self.__S:
                    continue
                
                # case 1: v has not been visited yet and has not been invited yet
                if v not in visited and v not in self.__invited_by_seed.keys():
                    # check if v accepts to take part to the auction
                    res = self.__invite(t, u, v, auction, prob, val)
                    
                    if res is not False:
                        # case 1.1: v accepts to take part to the auction
                        bids[v] = res[0]
                        reports[v] = res[1]
                        self.__invited_by_seed[v] = seed
                        if v not in self.__invited_by_nodes.keys():
                            self.__invited_by_nodes[v] = {u}
                        else:
                            self.__invited_by_nodes[v].add(u)
                        queue.append(v)
                    else:
                        # case 1.2: v does not accept to take part to the auction and refuses the invitation
                        if u in reports.keys() and v in reports[u]: 
                            reports[u].remove(v)    
                
                # case 2: v has not been visited yet and has been invited by another root
                elif v in self.__invited_by_seed.keys() and self.__invited_by_seed[v] != seed:
                    # case 2.1: spam
                    if u in reports.keys() and v in reports[u]: 
                        reports[u].remove(v)
                        # remove v from the auction
                        for elem in self.__invited_by_nodes[v]:
                            if elem in reports.keys() and v in reports[elem]:
                                reports[elem].remove(v)
                            if elem in bids.keys() and v in bids.keys():
                                del bids[v]

        return reports, bids
        
    def run(self, t, prob, val):
        
        self.__invited_by_seed.clear()
        self.__invited_by_nodes.clear()
        
        self.__S, auction = self.__init(t)

        bids = dict()
        reports = dict()
        
        revenue = 0
        
        for seed in self.__S:
            
            r, b = self.__build_reports_and_bids(seed, bids, reports, t, auction, prob, val)
            
            reports.update(r)
            bids.update(b)
            
            new_seller_net = set()
            for node in self.G[seed]:
                if node in bids.keys():
                    new_seller_net.add(node)
            
            allocation, payment = auction["auction"](self.k, new_seller_net, reports, bids)
            
            for all, pay in zip(allocation.values(), payment.values()):
                if all:
                    revenue += pay
                    
            bids.clear()
            reports.clear()
            
        return revenue
        
        
