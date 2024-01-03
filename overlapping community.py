import networkx as nx
from networks_gen import affiliationG

def overlapping_clustering(G, n_iter = 10):
    
    communities = {i: set() for i in range(n_iter)}
    affiliations = { i: {j: set() for j in range(n_iter)} for i in G.nodes() }
    
    for i in range(n_iter):
        communities[i] = set(frozenset(elem for elem in community) for community in nx.community.louvain_communities(G))
        l = 0
        for community in communities[i]:
            for index, node in enumerate(community):
                if node not in affiliations:
                    affiliations[node][i] = set()
                affiliations[node][i].add(l)
            l += 1
            
        print("Iteration", i, "communities:", l)
        
    # merge identical runs which returned the same communities
    for k, v in communities.items():
        for k1, v1 in communities.items():
            if k != k1 and v == v1:
                communities.pop(k1)
                
    # take as number of communities the most frequent one
    n = len(max(communities.values(), key=len))
    print(n)
    
    return communities, affiliations

if __name__ == '__main__':
    
    G = affiliationG(100, 4, 0.1, 2, 0.1, 1)
    communities, affiliations = overlapping_clustering(G)
    for comm in communities:
        print(communities[comm])
        