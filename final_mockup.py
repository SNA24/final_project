import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
from social_network_algorithms.mechanisms.SNCA import snca
from social_network_algorithms.mechanisms.VCG import vcg

import random

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
        
        self.__invited = dict() # {node: seed1, ...}

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
        return self.__auctions[2]

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
        
    def __build_reports_and_bids(self, u, bids, reports, t, auction, prob, val, visited=None, root=None):
        
        if root is not None:
            if u in self.__S:
                return reports, bids
        
        if visited is None:
            visited = set()
            root = u
            
        visited.add(u)
        
        for v in self.G[u]:
                
            if v in visited:
                continue
            
            if v not in self.__invited.keys() and v not in self.__S:
                res = self.__invite(t, u, v, auction, prob, val)
                if type(res) == tuple:
                    self.__invited[v] = root
                    bids[v] = res[0]
                    reports[v] = res[1]
                    reports, bids = self.__build_reports_and_bids(v, bids, reports, t, auction, prob, val, visited, root)
                elif u in reports.keys() and u != root and v in reports[u]:
                    reports[u].remove(v)
            else:
                if u in reports.keys() and u != root and v in reports[u]:
                    reports[u].remove(v)
                
        return reports, bids
        
    def run(self, t, prob, val):
        
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
        
        
