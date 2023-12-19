import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections

class Analyzer:
    
    def __init__(self, network):
        self.network = network
        
    def get_degree_distribution(self):
        "Returns the plot of the degree distribution with a log-log scale"
        
        # if the network is directed, we consider the in-degree
        if self.network.is_directed():
            degree_sequence = sorted([d for n, d in self.network.in_degree()], reverse=True)
        else:
            degree_sequence = sorted([d for n, d in self.network.degree()], reverse=True)
            
        degree_freq = {}
        for d in degree_sequence:
            if d in degree_freq:
                degree_freq[d] += 1
            else:
                degree_freq[d] = 1
                
        # Extract degrees and frequencies
        degrees = list(degree_freq.keys())
        frequencies = list(degree_freq.values())

        # Plotting in log-log scale
        plt.figure(figsize=(8, 6))
        plt.loglog(degrees, frequencies, marker='o', linestyle='None', color='b')
        plt.title('Log-log Degree Distribution')
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.grid(True)

        return plt
    
    def get_clustering_coefficient(self):
        pass
    
    def get_giant_component(self):
        pass
    
    def get_diameter(self):
        pass
        
if __name__ == '__main__':

    G = nx.Graph()  
    
    f = open('net_2', 'r')

    for line in f:
        line = line.split()
        G.add_edge(int(line[0]), int(line[1]))
        
    f.close()
            
    print(G.number_of_nodes())
    print(G.number_of_edges())
    
    plt = Analyzer(G).get_degree_distribution()
    plt.show()
    
    exit()
    
    