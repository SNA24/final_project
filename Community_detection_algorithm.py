#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy as CPY
import time as TM
import cdlib as CDL
import os

#similarity types:
    # MN : Meet of Neighbors
    # JC : Jaccard Coefficient
    # SI : Saltin Index
    # SO : Sorensen Index
    # HP : Hub Promoted Index
    # HD : Hub Depressed Index
    # CUSTOM : Custom Definition

def DETECT_COMMUNITIES(adj_list_name,stopCRITERION=0.01,similarity_type='HP'): 
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Define Required functions
    def READ_ADJACENCY_LIST_func(adj_list_name):
        DATA=[]
        with open(adj_list_name, 'r') as file:
            for line in file:
                LINE=[]
                for neigh in line.split(): LINE.append((neigh))
                DATA.append(LINE)
        dict1={} ; dict2={} ; i=0
        for line in DATA:
            dict2[i]=line[0] ; dict1[line[0]]=i
            i+=1
        GRAPH_adj_list=[]
        for line in DATA:
            GRAPH_adj_list.append([dict1.get(key) for key in line[1:]])
        return GRAPH_adj_list , dict2
    
    def nodes_similarity_calculator_func(var_adj,var_deg,var_type):
        var_allnodes_similarity=[]
        for var_node in range(len(var_adj)):
            var_eachnode_similarity_list=[]
            for var_neigh in var_adj[var_node]:
                var_sim=float(len(set(var_adj[var_node]) & set(var_adj[var_neigh])))
                if var_type=='MN': var_sim=var_sim/1
                if var_type=='SI': var_sim=var_sim/np.sqrt(var_deg[var_node]*var_deg[var_neigh]) 
                if var_type=='SO': var_sim=2.0*var_sim/(var_deg[var_node]+var_deg[var_neigh])
                if var_type=='JC': var_sim=var_sim/(var_deg[var_node]+var_deg[var_neigh]-var_sim)
                if var_type=='HP': var_sim=var_sim/min(var_deg[var_node],var_deg[var_neigh])
                if var_type=='HD': var_sim=var_sim/max(var_deg[var_node],var_deg[var_neigh])
                #if var_type=='CUSTOM': var_sim= Custom Definition
                var_eachnode_similarity_list.append(var_sim)
            var_allnodes_similarity.append(var_eachnode_similarity_list)
        return var_allnodes_similarity
           
    def U1_func(var_adj,var_S,var_similarity,var_deg,var1):
        var_pay=0
        for i in range(var_deg[var1]):
            if var_S[var1]==var_S[var_adj[var1][i]]:
                var_pay+= (  var_similarity[var1][i]  +1) 
        return var_pay
    
    def U2_func(var_adj,var_S,var_similarity,var_deg,var1):
        var_pay=0 
        for i in range(var_deg[var1]):
            var_w=( var_similarity[var1][i]  + 1 ) 
            shared=len(set(var_S[var1]) & set(var_S[var_adj[var1][i]]))
            if shared>0:
                var_pay+= var_w / (len(set(var_S[var_adj[var1][i]])))**(0.5)
        return var_pay
    
    def thresh_calculator_func(var_MB):
        #var_MB : membership coefficients 
        return np.sqrt(np.mean(np.array(var_MB)**2))
    
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Start_time=TM.time()                        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<START
    GRAPH_adj, dict2 = READ_ADJACENCY_LIST_func(adj_list_name)
    DEG=[]
    for nodesneigh in GRAPH_adj: DEG.append(len(nodesneigh))
    DEG=np.array(DEG)  #nodes degree list
    n=len(GRAPH_adj) #total number of nodes
    m=0.0
    for node_neigh in GRAPH_adj: m+=len(node_neigh)
    m=m/2    #total number of edges
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Initialization
    nodes_similarity=nodes_similarity_calculator_func(GRAPH_adj,DEG,similarity_type) 
    S=np.zeros([n],dtype=int)
    for i in range(n):
        S[i]=i                                 
    fix=0 ; Sprev=S+0 ; Iteration=0 
    diff=np.sum(S == Sprev)
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Phase2
    while fix<1:
        Iteration+=1
        for step in range(n):
            node=(step+0)%n
            neighs=GRAPH_adj[node] ; U_List=[] ; G1=[S[node]]
            for i in neighs:
                G1.append(S[i])
            G1=list(set(G1))
            for i in G1:
                saveparam=S[node]+0
                S[node]=i+0
                U=U1_func(GRAPH_adj,S,nodes_similarity,DEG,node)
                S[node]=saveparam+0
                U_List.append(U)
            select=G1[U_List.index(max(U_List))]
            S[node]=select
        if abs(np.sum(S == Sprev)-diff) <= int(stopCRITERION*n) : fix+=1
        else: fix=0
        diff=np.sum(S == Sprev) ; Sprev=S+0
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Phase2
    SS=[]
    for i in S:
        SS.append([i])
    phase2_repeat=1
    while phase2_repeat<=2:
        COM_ID=[]     # Community IDs which nodes belong to
        MEMB_COE=[]    # membership coefficient of nodes
        thresh=[]
        for node in range(n):
            neighs=GRAPH_adj[node] ; U_List=[] ; G1=CPY.copy(SS[node])
            for i in neighs:
                G1+=SS[i]
            G1=list(set(G1))
            for i in G1: 
                saveparam=SS[node]+[]
                SS[node]=[i]
                U=U2_func(GRAPH_adj,SS,nodes_similarity,DEG,node)
                SS[node]=saveparam+[]
                U_List.append(U)
            U_List=np.array(U_List)
            memb_coe=U_List/max(U_List) if len(U_List)>0 else []
            com_id=G1
            MEMB_COE.append(memb_coe)
            COM_ID.append(com_id)
            thresh.append( thresh_calculator_func(memb_coe) )
        crisp_comms_dict={}
        nodes_total_membership=np.zeros(n)
        SS=[]
        for node in range(n):
            newS=[]
            for i in range(len(MEMB_COE[node])):
                if MEMB_COE[node][i]>=thresh[node]:
                    try: crisp_comms_dict[COM_ID[node][i]]+=[node]
                    except: crisp_comms_dict[COM_ID[node][i]]=[node]
                    nodes_total_membership[node]+=1
                    newS.append(COM_ID[node][i])
            SS.append(newS)
        phase2_repeat+=1
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    crisp_comms=[]
    for i in crisp_comms_dict:
        crisp_comms.append(crisp_comms_dict[i]) 
    COMMUNITY_STRUCTURE=[]
    for line in crisp_comms:
        COMMUNITY_STRUCTURE.append([dict2.get(key) for key in line])
        
    End_time=TM.time()
    Execution_time=End_time-Start_time      #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<END
    
    #>>>>>>>>>>>>>>>>>>>Saving Results
    try:name=adj_list_name[0:adj_list_name.index('.')]
    except: name=adj_list_name
    try: os.remove('./%s_communities.txt'%name)
    except: pass
    with open('./%s_communities.txt'%name, 'a') as the_file:
        for community in COMMUNITY_STRUCTURE:
            for node in community:
                the_file.write(' %s'%node)
            the_file.write('\n')
    try: os.remove('./%s_Iterations.txt'%name)
    except: pass
    with open('./%s_Iterations.txt'%name, 'a') as the_file: the_file.write('Number of Iterations = %d'%Iteration)
    try: os.remove('./%s_Execution_Time.txt'%name)
    except: pass
    with open('./%s_Execution_Time.txt'%name, 'a') as the_file: the_file.write('Execution Time = %.4f s'%Execution_time)
    #>>>>>>>>>>>>>>>>>>>End of Saving Results
    
    return COMMUNITY_STRUCTURE, Iteration, Execution_time

#%%
from networks_gen import affiliationG
import networkx as nx
# G = affiliationG(100, 6, 0.1, 2, 0.1, 1)
# save adjacency list to file
# nx.write_adjlist(G, 'prova.txt')
DETECT_COMMUNITIES('prova.txt')