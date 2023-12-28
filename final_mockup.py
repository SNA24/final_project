import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
import networkx as nx
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
            }
        ]
        self.__invited = dict()
        
        # self.__invided = {indited_node: {inviting_root}, ...}

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
        return self.G.nodes()
    
    def __choose_auction(self):
        return self.__auctions[0]

    def __init(self, t):
        return self.__choose_S(), self.__choose_auction()

    def __invite(self, t, u, v, auction, prob, val):
        if prob(u, v, t):
            bid_v = val(t, v)
            S_v = self.G[v]
            if not auction["truthful_bidding"]:
                bid_v = random.randint(0.5*bid_v, bid_v)
            if not auction["truthful_reporting"]:
                S_v = {w for w in S_v if random.random() <= 0.2}
            return bid_v, S_v
        else:
            return False
        
    def __build_report_and_bid(self, S, t, u, auction, prob, val):
        
        bids = dict()
        reports = dict()
        
        for v in S:
            
            bid_v, S_v = self.__invite(t, u, v, auction, prob, val)
            if bid_v:
                bids[v] = bid_v
                reports[v] = S_v
                ret_bids, ret_reports = self.__build_report_and_bid(S_v, t, v, auction, prob, val)
                bids.update(ret_bids)
                reports.update(ret_reports)
                
        return reports, bids

    def run(self, t, prob, val):
        S, auction = self.__init(t)
        
        bids = dict()
        reports = dict()
        
        revenue = 0
        
        for s in S:
            seller_net = self.G[s]
            reports[s], bids[s] = self.__build_report_and_bid(seller_net, t, s, auction, prob, val)
            allocation, payment = auction["auction"](self.k, seller_net, reports[s], bids[s])
            for all, pay in zip(allocation.values(), payment.values()):
                if all:
                    revenue += pay
            
        return revenue
        
        
