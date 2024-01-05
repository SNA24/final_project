import networkx as nx
import random
from networks_gen import affiliationG 
from final_mockup import SocNetMec
from tqdm import tqdm

prob = dict()

def input_data():
    n = 1000
    G = affiliationG(n, 5, 0.5, 1, 0.1, 3)
    # G = nx.read_edgelist("test")
    # save the graph
    # nx.write_edgelist(G, "test")
    print("Graph generated")
    # k = random.randint(int(n * 0.1), int(n*0.5))
    # T = random.randint(20000, 100000)
    k = 5
    T = 5000

    #for the oracle val
    val = dict()
    for t in range(T):
        val[t] = dict()
        for u in G.nodes():
            val[t][u] = random.randint(1, 50)
    
    #for the oracle prob
    p = dict()
    
    for u in G.nodes():
        p[u] = dict()
        
    for u in G.nodes():
        for v in G[u]:
            if v not in p[u]:
                t=min(0.25, 10/max(G.degree(u), G.degree(v)))
                p[u][v] = p[v][u] = random.uniform(0, t)
            
    return G, k, T, val, p

def probf(u, v, t):
    if (u, v, t) in prob.keys():
        return prob[(u, v, t)]
    r = random.random()
    if r <= p[u][v]:
        prob[(u, v, t)] = True
        return True
    prob[(u, v, t)] = False
    return False

def valf(t, u):
    return val[t][u]

G, k, T, val, p = input_data()
snm=SocNetMec(G, T, k)
revenue = 0
for step in tqdm(range(T)):
    revenue += snm.run(step, probf, valf)

print(revenue)