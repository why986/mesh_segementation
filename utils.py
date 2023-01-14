import heapq
from flow import FlowGraph
from collections import defaultdict
import numpy as np

def calc_shortest_dist(N, adj_csr):
    # N: number of nodes 
    # adj_csr: adjacency matrix of the graph, in csr format
    # return: the shortest distance matrix
    dis = []
    for S in range(N):
        # dijkstra
        dis_S = defaultdict(lambda:float("inf"))
        dis_S[S] = 0
        q = [(0, S)]
        flag = set()
        while q:
            _, u = heapq.heappop(q)
            if u in flag:
                continue
            for j in range(adj_csr.indptr[u], adj_csr.indptr[u+1]):
                v = adj_csr.indices[j]
                w = adj_csr.data[j]
                if dis_S[v] > dis_S[u] + w:
                    dis_S[v] = dis_S[u] + w
                    heapq.heappush(q, (dis_S[v], v))
        dis.append(dis_S)
    return dis

def check_seeds(seeds, dis, eps):
    seeds_dis_list = []
    for i in seeds:
        for j in seeds:
            if i != j:
                seeds_dis_list.append(dis[i][j])
    return np.mean(seeds_dis_list) < eps



def build_flow_graph(ang_dist_mat, fuzzy_faces, S1, S2, color, faces):
    # build flow graph
    flow_graph = FlowGraph()
    A1 = set()
    A2 = set()
    # calculate the mean angle distance 
    ang_dist_sum = 0.
    cnt = 0
    for u in fuzzy_faces:
        for j in range(ang_dist_mat.indptr[u], ang_dist_mat.indptr[u+1]):
            v = ang_dist_mat.indices[j]
            if not v in faces:
                continue
            if color[v] == color[S1]:
                A1.add(v)
                ang_dist_sum += ang_dist_mat.data[j]
                cnt += 1
            elif color[v] == color[S2]:
                A2.add(v)
                ang_dist_sum += ang_dist_mat.data[j]
                cnt += 1
            elif v in fuzzy_faces:
                ang_dist_sum += ang_dist_mat.data[j]
                cnt += 1
    ang_dist_mean = ang_dist_sum / cnt

    # add edges
    for u in fuzzy_faces:
        for j in range(ang_dist_mat.indptr[u], ang_dist_mat.indptr[u+1]):
            v = ang_dist_mat.indices[j]
            if not v in faces:
                continue
            c = ang_dist_mat.data[j]
            c = 1 / (1 + c / ang_dist_mean)
            if color[v] == color[S1]:
                flow_graph.add_edge(v, u, c)
            elif color[v] == color[S2]:
                flow_graph.add_edge(u, v, c)
            elif v in fuzzy_faces:
                flow_graph.add_edge(u, v, c)
    
    for u in A1:
        flow_graph.add_edge(flow_graph.S, u, float("inf"))
    for u in A2:
        flow_graph.add_edge(u, flow_graph.T, float("inf"))
    return flow_graph


                