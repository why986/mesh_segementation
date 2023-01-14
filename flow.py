import numpy as np
import queue
from collections import *

class FlowEdge():
    def __init__(self, u, v, c, f):
        # u: source v: destination c: capacity f: flow
        self.u = u
        self.v = v
        self.c = c
        self.f = f

class FlowGraph():
    def __init__(self):
        self.edges = [] # list of edges
        self.graph = defaultdict(list) # edges of each vertex
        self.p = {}
        self.cnt = 0 # number of edges
        self.S = -1
        self.T = -2

    def add_edge(self, u, v, c):
        self.edges.append(FlowEdge(u, v, c, 0))
        self.graph[u].append(self.cnt)
        self.cnt += 1

        self.edges.append(FlowEdge(v, u, 0, 0))
        self.graph[v].append(self.cnt)
        self.cnt += 1
        
    def calc_max_flow(self):
        flow = 0
        S, T = self.S, self.T
        while True:
            a = defaultdict(int)
            Q = queue.Queue()
            Q.put(S)
            a[S] = float("inf")
            while Q.empty() == False:
                x = Q.get()
                for idx in self.graph[x]:
                    e = self.edges[idx]
                    if a[e.v] == 0 and e.c > e.f:
                        self.p[e.v] = idx
                        a[e.v] = min(a[x], e.c - e.f)
                        Q.put(e.v)
                if a[T] != 0:
                    break
            if a[T] == 0:
                break
            u = T
            while True:
                if u == S:
                    break
                self.edges[self.p[u]].f += a[T]
                self.edges[self.p[u] ^ 1].f -= a[T]
                u = self.edges[self.p[u]].u
            flow += a[T]
        return flow

    def get_result(self):
        searched_v = set()
        Q = queue.Queue()
        Q.put(self.S)
        searched_v.add(self.S)
        while Q.empty() == False:
            x = Q.get()
            for idx in self.graph[x]:
                e = self.edges[idx]
                if idx % 2 == 0 and e.c > e.f and e.v not in searched_v:
                    searched_v.add(e.v)
                    Q.put(e.v)
        not_searched = []
        for v in self.graph:
            if v not in searched_v:
                not_searched.append(v)
        return list(searched_v), not_searched
        

