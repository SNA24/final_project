import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
import math
from lesson5 import affiliationG 

# Implementation of the BigCLAM algorithm described in the
# "Community-Affiliation Graph Model for Overlapping Network 
# Community Detection" paper by Stanford University,
# also some videos by Stanford University were used as reference

def log_likelihood(u, F, G, sum_v, i):
    """Computes the log-likelihood of the node u in the graph G with respect to the affiliation matrix F."""
    
    sum_1 = np.zeros(i)
    for v in G.neighbors(u):
        dot_product = np.dot(F[u], np.transpose(F[v]))
        denominator = 1 - math.exp(-dot_product)
        sum_1 += (F[v] * np.exp(-dot_product) / denominator)

    sum_2 = sum_v - F[u] - np.sum(F[list(G.neighbors(u))], axis=0)

    return sum_1 - sum_2

def coordinate_gradient_ascent(log_likelihood, F, G, sum_v, learning_rate, max_iter):
    """Computes the coordinate gradient ascent algorithm for the log-likelihood function."""
    
    n, m = F.shape

    while max_iter > 0:
        
        tot_gradient = np.zeros(m)
        new_F = np.zeros((n, m))
        
        for u in range(n):
            gradient_u = log_likelihood(u, F, G, sum_v, m)
            tot_gradient += gradient_u
            new_F[u] = F[u] + learning_rate * gradient_u
            new_F[u] = np.maximum(new_F[u], 0)
            
        F = new_F
        sum_v = np.sum(F, axis=0)
        
        max_iter -= 1
        
        if np.sum(np.abs(tot_gradient)) < 0.0001:
            break
        
    return F

def compute_c(G, m):
    """Computes the number of communities c and the affiliation matrix F."""
    
    n = len(G.nodes())
    learning_rate = 0.0001
    max_iter = 1000

    F = np.random.rand(n, m)
    sum_v = np.sum(F, axis=0)
    
    F = coordinate_gradient_ascent(log_likelihood, F, G, sum_v, learning_rate, max_iter)
    
    mean = np.mean(F)

    affiliations, communities = {}, {}
    for u in range(n):
        affiliations[u] = [i for i in range(m) if F[u][i] > mean]
        for i in affiliations[u]:
            if i not in communities:
                communities[i] = [u]
            else:
                communities[i].append(u)

    # compute the number of elements > 0.01 per-row and make the mean
    mean_length = np.mean(np.sum(F > 0.0099, axis=1))
    
    return math.ceil(mean_length), affiliations, communities

def compute_p(G, affiliations, communities):
    """Computes the probability p of a node to have an edge towards a community."""

    num_strong_ties_per_node = [degree-1 for _, degree in G.degree()]
    
    # compute the number of edges each nodes has with nodes in the same community
    num_strong_ties_per_community = {}
    for u in affiliations:
        num_strong_ties_per_community[u] = {}
        for i in affiliations[u]:
            for neighbor in G.neighbors(u):
                if i in affiliations[neighbor]:
                    if i not in num_strong_ties_per_community[u]:
                        num_strong_ties_per_community[u][i] = 1
                    else:
                        num_strong_ties_per_community[u][i] += 1
                    
    # compute the probability p of a node to have an edge towards a community
    p = {}
    for u in num_strong_ties_per_community:
        p[u] = {}
        for i in num_strong_ties_per_community[u]:
            p[u][i] = num_strong_ties_per_community[u][i] / sum(len(communities[i]) for i in communities)
            
    mean_p = np.mean([p[u][i] for u in p for i in p[u]])
    
    return mean_p

def compute_degree_distance(G1, G2):
    """Computes the degree distance between two graphs G1 and G2."""

    degree_distribution_G1 = [d for _, d in G1.degree()]
    degree_distribution_G2 = [d for _, d in G2.degree()]
        
    return sum([abs(degree_distribution_G1[i] - degree_distribution_G2[i]) for i in range(len(degree_distribution_G1))])
    
if __name__ == '__main__':

    best_dist = float('inf')

    early_stopping = 5
    
    G = affiliationG(100, 5, 0.75, 4, 0.05, 1)
    
    n = len(G.nodes())
    m = range(4, 6)
    q = [0.1 * n for n in range(1, 10)]
    s = 1

    for i in m:
        
        c, affiliations, communities = compute_c(G, i)
        
        p = compute_p(G, affiliations, communities)

        for j in q:
                
            G2 = affiliationG(n, i, j, c, p, s)
            
            dist = compute_degree_distance(G, G2)
            print("m = {}, q = {}, c = {}, p = {}, dist = {}".format(i, j, c, p, dist))
            
            if dist < best_dist:
                best_dist = dist
                best_params = [i, j, c, p]
                
            if dist < 0.001 or early_stopping == 0:
                early_stopping = 5
                break
            
            elif dist > best_dist and early_stopping > 0:
                early_stopping -= 1
                        
    print(best_params, best_dist)