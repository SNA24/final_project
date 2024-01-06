import networkx as nx
from networks_gen import affiliationG

def calculate_modularity(graph, communities):
    
    num_edges = graph.number_of_edges()
    adjacency_matrix = nx.to_numpy_array(graph)
    modularity = 0.0

    for community_i in communities.values():
        for node_i in community_i:
            node_i -= 1
            for community_j in communities.values():
                for node_j in community_j:
                    node_j -= 1
                    modularity += (adjacency_matrix[node_i, node_j] - 
                                   (graph.degree[node_i] * graph.degree[node_j]) / (2 * num_edges)) * (
                                      int(community_i == community_j) - 
                                      (graph.degree[node_i] * graph.degree[node_j]) / (2 * num_edges)
                                   )

    modularity /= (2 * num_edges)
    
    return modularity

def operlapping_louvain(G, n_iter = 2, lambda_ = 0, threshold = 0.99):
    
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
                        
    # save the belonging matrix in a file
    belonging_matrix_file = open("belonging_matrix.txt", "a")
    for row in belonging_matrix:
        belonging_matrix_file.write(str(row) + "\n")
    belonging_matrix_file.close()
                          
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
            
    print("Number of communities:", len(communities))
            
    # STEP 3. merge the communities which have the same nodes or are subsets of other communities
    for k, v in communities.copy().items():
        for k1, v1 in communities.copy().items():
            if len(v.intersection(v1)) > 0 and k != k1:
                communities.pop(k1)
                print("Removed community", k1)
            # elif check if they have a lot of nodes in common and if yes merge them
            else:
                print("Checking if community", k, "and", k1, "have a lot of nodes in common")
                # compute jaccard similarity
                intersection = len(v.intersection(v1))
                union = len(v.union(v1))
                jaccard_similarity = intersection / union
                if jaccard_similarity > threshold and k != k1:
                    communities[k] = v.union(v1)
                    communities.pop(k1)

    return communities

if __name__ == "__main__" :
    
    # G = affiliationG(5000, 1, 0.1, 1, 0.1, 1)
    # print("Number of nodes:", G.number_of_nodes())
    #clustering_coefficient = nx.average_clustering(G)
    # print("Clustering coefficient:", clustering_coefficient)
    G = nx.read_edgelist("net_2", nodetype = int)
    communities = operlapping_louvain(G)
    print("Number of communities:", len(communities))

    


        