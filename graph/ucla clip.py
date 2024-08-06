import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools
from graph.infogcn import tools as tools_infogcn

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_part = 5
self_link_part = [(i, i) for i in range(num_part)]
inward_part = [(0, 1), (0, 2), (0, 3), (0, 4)]
outward_part = [(j, i) for (i, j) in inward_part]

symmetry = [(0, 0),(1, 1), (2, 2), (3, 3), (4, 8),(5, 9), (6, 10), 
            (7, 11),(12, 16),(13, 17),(14, 18),(15, 19)]
left_index = [i for (i, j) in symmetry if i != j]
right_index = [j for (i, j) in symmetry if i != j]
middle_index = [i for (i, j) in symmetry if i == j]

torso = np.array([ 1,  2,  3,  4]) - 1                 # torso
arm   = np.array([ 5,  6,  7,  8,  9, 10, 11, 12]) - 1 # arm
leg   = np.array([13, 14, 15, 16, 17, 18, 19, 20]) - 1  # leg

# parts = [
#             np.array([1, 2, 3, 4]) - 1,      # torso
#             np.array([5, 6, 7, 8]) - 1,     # left_arm
#             np.array([9, 10, 11, 12]) - 1,  # right_arm
#             np.array([13, 14, 15, 16]) - 1,  # left_leg
#             np.array([17, 18, 19, 20]) - 1,  # right_leg           
#         ]
# joints = np.array([16,17,18,19,
#                     0,1,2,3,
#                     4,5,6,7,
#                     8,9,10,11,
#                     12,13,14,15])

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.symmetry_info = (num_node, left_index, right_index, torso, arm, leg)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_part = tools.get_spatial_graph(num_part, self_link_part, inward_part, outward_part)
        self.A_outward_binary = tools_infogcn.get_adjacency_matrix(self.outward, self.num_node)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
