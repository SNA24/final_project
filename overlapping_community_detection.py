import networkx as nx
from networks_gen import affiliationG
import math

def operlapping_louvain(G, n_iter = 20, lambda_ = 0.6):
    
    # STEP 1. execute the louvain algorithm n_iter times and record the results
    results = []
    for i in range(n_iter):
        results.append([])
        communities = nx.community.louvain_communities(G)   
        for community in communities:
            results[i].append(community)

    # STEP 2. build the belonging matrix with N rows (number of nodes) and C columns (number of communities)
    n_nodes = len(G.nodes())
    
    # initialize the belonging matrix, each node ia a community
    belonging_matrix = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]

    # fill the belonging matrix
    
    for row in results:
        for i, community in enumerate(row):
            for node in community:
                for neighbor in G.neighbors(node):
                    if neighbor in community:
                        belonging_matrix[node][i] += 1
                          
    # normalize the belonging matrix in range [0,1] to compute the belonging coefficient of each node to each community
    # take the maximum coefficient of the whole matrix
    max_coefficient = max([max(row) for row in belonging_matrix])
    for row in belonging_matrix:
        for i in range(len(row)):
            row[i] /= max_coefficient
            
    # compute lambda_ as mean of the coefficients of the belonging matrix
    lambda_ = sum([sum(row) for row in belonging_matrix]) / (n_nodes * n_nodes)
    
    # if the coefficient is greater than lambda_ then the node belongs to the community
    communities = {i: set() for i in range(n_nodes)}
    i = 1
    for node in belonging_matrix:
        for j in range(n_nodes):
            if node[j] >= lambda_:
                communities[j].add(i)
        i += 1
        
    # remove empty communities
    for community in list(communities.keys()):
        if len(communities[community]) == 0:
            communities.pop(community)
    
        
    return communities

if __name__ == "__main__":
    
    G = affiliationG(100, 4, 0.02, 1, 0.1, 1)
    # G = nx.read_edgelist("net_2", nodetype=int)
    communities = operlapping_louvain(G)
    for community in communities:
        print(community)
        