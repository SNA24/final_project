import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from social_network_algorithms.mechanisms.MUDAN import mudan
from social_network_algorithms.mechanisms.SNCA import snca
from social_network_algorithms.mechanisms.VCG import vcg

from multi_armed_bandit import UCB_Learner, EpsGreedy_Learner, Exp_3_Learner
import networkx as nx

import random, math

class SocNetMec:
    
    def __init__(self, G, T, k):
        
        self.G = G
        self.T = T
        self.k = k
        
        # the following auctions are implemented in the social_network_algorithms.mechanisms package
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
        
        # we decide to use the UCB learner because we experimentally saw that on the average it has the best performance, even if Exp3 would have been
        # better suited for the problem of the unknown environment (probabilities and valuations changing at each step)
        # other learners are supported too, but they are not used in the final test
        self.__learner = UCB_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        # self.__learner = EpsGreedy_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        # self.__learner = Exp_3_Learner(self.G, [auction["name"] for auction in self.__auctions], self.T)
        
        # the learner during its initialization ranks the available nodes according to a centrality measure (in this case a paralell version of PageRank
        # which can be found in the social_network_algorithms.centrality_measures package) and then chooses the first n nodes in the ranking as seeds, whose 
        # default value is 4, but it is suggested to use a smaller value if the graph is large and the available memory is limited
        ranking = self.__learner.get_ranking()
        # we cache the bfs trees because they are frequently used in the run method in order to make the computations only at the beginning and make the
        # algorithm terminate in a reasonable amount of time
        # obviously we do it under the assumption that the length of the seeds' set B we choose is small enought to not use too much space in memory to store
        # the BFS trees
        self.cache_bfs = {n: nx.bfs_tree(self.G, n) for n in ranking}
        
        self.__spam = set()
        
    def give_eps(self, G, t):
        # this method is only used when using the eps-greedy learner
        if t == 0:
            return 1  #for the first step we cannot make exploitation, so eps_1 = 1
        return (len(G.nodes())*math.log(t+1)/(t+1))**(1/3)
        
    def __init(self, t):
        """
        This method takes a timestep t and returns the set of seeds S and the auction chosen by the learner
        """
        if type(self.__learner) == UCB_Learner or type(self.__learner) == Exp_3_Learner:
            arms, auction = self.__learner.play_arm()
        else:
            arms, auction = self.__learner.play_arm(self.give_eps(self.G, t))
        arms = set(arms)
        for a in self.__auctions:
            if a["name"] == auction:
                break
        return arms, a
    
    def __find_reachable_nodes(self, S):
        """
        This method takes a set of nodes S and returns a list of BFS trees, one for each node in S
        """
        self.__spam.clear()

        if len(S) < 2:
            return [self.cache_bfs[list(S)[0]]]
                    
        trees = []
        
        for index, s in enumerate(list(S)):
            tree = self.cache_bfs[s]
            if index == 0:
                self.__spam = set(tree.nodes())
            else:
                # the nodes in spam are the ones which can be reached by more than one seed and have not to be considered in the 
                # information spreading process
                self.__spam = self.__spam.intersection(tree.nodes())
            trees.append(tree)
            
        return trees

    def __invite(self, t, u, v, auction, prob, val):
        """
        This method takes a timestep t, two nodes u and v, an auction, a probability function prob and a valuation function val and returns a tuple
        (bid_v, S_v) if the invitation is accepted, False otherwise
        """
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
        
    def build_reports_and_bids(self, t, auction, prob, val, tree):
        """
        This method takes a timestep t, an auction, a probability function prob, a valuation function val and a BFS tree and returns two dictionaries:
        reports and bids
        - reports is a dictionary that maps each node v in the BFS tree to the set of nodes that v can invite
        - bids is a dictionary that maps each node v in the BFS tree to the bid that v will make
        """
        
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
            for w in reports:
                if v in reports[w]:
                    reports[w].remove(v)
            
        return reports, bids
        
    def run(self, t, prob, val):
        """
        This method takes a timestep t, a probability function prob and a valuation function val and returns the revenue obtained at timestep t
        - it first initializes the set of seeds S and the auction
        - then it builds the reports and the bids for each node in S
        - then it builds the seller network for each node in S and runs the auction on it
        - finally it computes the revenue and returns it
        """
        
        self.__S, auction = self.__init(t)
        
        bids = {s: {} for s in self.__S}
        reports = {s: {} for s in self.__S}
        
        revenue = 0
        trees = self.__find_reachable_nodes(self.__S)
        
        if len(self.__spam) != len(self.G.nodes()):
            for index, seed in enumerate(self.__S):
                reports[seed], bids[seed] = self.build_reports_and_bids(t, auction, prob, val, trees[index])

        def build_seller_net_and_run_auction(seed):
            new_seller_net = set()
            for node in self.G[seed]:
                if node in bids[seed].keys():
                    new_seller_net.add(node)
            allocation, payment = auction["auction"](self.k, new_seller_net, reports[seed], bids[seed])
            return allocation, payment
                
        allocation = {}
        payment = {}
        
        for seed in self.__S:
            
            allocation, payment = build_seller_net_and_run_auction(seed)
            
            for node in allocation:
                revenue += payment[node]
                
            allocation.clear()
            payment.clear()
                    
        bids.clear()
        reports.clear()
            
        self.__learner.receive_reward(revenue)
        print("Timestep: ", t, "Revenue: ", revenue)
            
        return revenue
        
        
