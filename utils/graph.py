import sys
import numpy as np

sys.path.extend(['../'])

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

num_node_kv2 = 25
self_link_kv2 = [(i, i) for i in range(num_node_kv2)]
inward_ori_index_kv2 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_kv2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_kv2]
outward_kv2 = [(j, i) for (i, j) in inward_kv2]
neighbor_kv2 = inward_kv2 + outward_kv2

class Graph_kv2:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node_kv2
        self.self_link = self_link_kv2
        self.inward = inward_kv2
        self.outward = outward_kv2
        self.neighbor = neighbor_kv2

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node_kv2, self_link_kv2, inward_kv2, outward_kv2)
        else:
            raise ValueError()
        return A


num_node_kv3 = 32
self_link_kv3 = [(i, i) for i in range(num_node_kv3)]
inward_kv3 = [(0, 1), (1, 2), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 7),
          (11, 2), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15), (17, 14),
          (18, 0), (19, 18), (20, 19), (21, 20), (22, 0), (23, 22), (24, 23), (25, 24),
          (26, 3), (27, 26), (28, 27), (29, 28), (30, 27), (31, 30)]
outward_kv3 = [(j, i) for (i, j) in inward_kv3]
neighbor_kv3 = inward_kv3 + outward_kv3

class Graph_kv3:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node_kv3
        self.self_link = self_link_kv3
        self.inward = inward_kv3
        self.outward = outward_kv3
        self.neighbor = neighbor_kv3

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node_kv3, self_link_kv3, inward_kv3, outward_kv3)
        else:
            raise ValueError()
        return A


num_node_as = 19
self_link_as = [(i, i) for i in range(num_node_as)]
inward_as = [(0, 9), (1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (10, 9),
          (11, 9), (12, 11), (13, 12), (14, 13), (15, 9), (16, 15), (17, 16), (18, 17)]
outward_as = [(j, i) for (i, j) in inward_as]
neighbor_as = inward_as + outward_as

class Graph_asian:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node_as
        self.self_link = self_link_as
        self.inward = inward_as
        self.outward = outward_as
        self.neighbor = neighbor_as

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node_as, self_link_as, inward_as, outward_as)
        else:
            raise ValueError()
        return A