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
        
        self.__cache = dict() # {S1: {spam}, ...}
        self.__spam = set() # {node, ...}

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
            
    def __find_reachable_nodes(self, S):
        
        if frozenset(S) in self.__cache:
            self.__spam = self.__cache[frozenset(S)]
            return
        
        self.__spam = set()
        visited = dict()
        for s in S:
            visited[s] = set()
            # make a bfs from s
            queue = [s]
            while queue:
                node = queue.pop(0)
                if node not in visited[s]:
                    visited[s].add(node)
                    queue.extend(self.G[node])
                other = [i for i in S if i != s]
                for o in other:
                    if o in visited.keys() and node in visited[o]:
                        self.__spam.add(node)
                        break
                    
        if frozenset(S) not in self.__cache:
            self.__cache[frozenset(S)] = self.__spam

    def __choose_S(self):
        # returns a subsets S of G's nodes according to some criteria
        S = set()
        while len(S) < 5:
            S.add(random.choice(list(self.G.nodes())))
        
        self.__find_reachable_nodes(S)
        
        return S
    
    def __choose_auction(self):
        return self.__auctions[1]

    def __init(self, t):
        return self.__choose_S(), self.__choose_auction()

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u, v, t):
            bid_v = val(t, v)
            S_v = set(self.G[v]).difference(self.__spam).difference(self.__S)
            if not auction["truthful_bidding"]:
                bid_v = random.randint(0.5*bid_v, bid_v)
            if not auction["truthful_reporting"]:
                S_v = {w for w in S_v if random.random() <= 0.2}
            return bid_v, S_v
        else:
            return False
        
    def __build_reports_and_bids(self, bids, reports, S, t, u, auction, prob, val, visited=None):
        
        if visited is None:
            visited = set()

        if len(S) == 0 or u in visited:
            return reports, bids
        
        visited.add(u)
        
        for v in S.copy():
            
            if v in visited or v in self.__S:
                continue
            
            res = self.__invite(t, u, v, auction, prob, val)
            
            if type(res) == bool:
                reports[u].remove(v)
                continue
            
            bid_v, S_v = res
            
            bids[v] = bid_v
            reports[v] = S_v
            
            reports, bids = self.__build_reports_and_bids(bids, reports, S_v, t, v, auction, prob, val, visited)
            
        return reports, bids
        
    def run(self, t, prob, val):
        
        self.__S, auction = self.__init(t)

        bids = dict()
        reports = dict()
        
        revenue = 0
        
        for s in self.__S:

            seller_net = self.G[s]
            new_seller_net = set(seller_net)
            
            for neighbor in seller_net:
                
                if neighbor in self.__spam or neighbor in self.__S:
                    new_seller_net.remove(neighbor)
                    continue
                
                res = self.__invite(t, s, neighbor, auction, prob, val)
                
                if type(res) == bool:
                    new_seller_net.remove(neighbor)
                else:
                    bid_v, S_v = res
                    bids[neighbor] = bid_v
                    reports[neighbor] = S_v
            
            report, bid = self.__build_reports_and_bids(bids, reports, new_seller_net, t, s, auction, prob, val)
            
            bids.update(bid)
            reports.update(report)
            
            allocation, payment = auction["auction"](self.k, new_seller_net, reports, bids)
            
            for all, pay in zip(allocation.values(), payment.values()):
                if all:
                    revenue += pay
                    
            bids.clear()
            reports.clear()
            
        return revenue
        
        
