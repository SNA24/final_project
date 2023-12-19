import networkx as nx
import random

class SocNetMec:
    
    def __init__(self, G, T, k):
        self.G = G
        self.T = T
        self.k = k

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

    #MOCK-UP IMPLEMENTATION: It returns at each time step the first node
    def __init(self, t):
        for u in self.G.nodes():
            return u, __mock_auction

    #MOCK-UP IMPLEMENTATION: It returns bid 1 and no report
    def __invite(t, u, v, auction, prob, val):
        if prob(u,v):
            return 1, set()
        else:
            return False

    #NOT IMPLEMENTED. It simply adds to the revenue a random integer
    def run(self, t, prob, val):
        return random.randint(1, 10)
