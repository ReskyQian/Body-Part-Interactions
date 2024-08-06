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

symmetry = [(0, 0),(1, 1), (2, 2), (3, 3), (4, 8),(5, 9), (6, 10), 
            (7, 11),(12, 16),(13, 17),(14, 18),(15, 19)]
left_index = [i for (i, j) in symmetry if i != j]
right_index = [j for (i, j) in symmetry if i != j]
middle_index = [i for (i, j) in symmetry if i == j]

# joint
torso1 = np.array([2,3,0,1,4,8,12,16])          # torso
arm1  = np.array([10,11,6,7,8,9,4,5])          # arm
leg1  = np.array([16,17,18,19,12,13,14,15])    # leg

# bone
torso2 = np.array([2,0,1,7,3,11,15])              # torso
arm2  = np.array([7,8,9,10,3,4,5,6])             # arm
leg2   = np.array([11,12,13,14,15,16,17,18])      # leg

class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.joint_info = (torso1, arm1, leg1)
        self.bone_info = (torso2, arm2, leg2)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools_infogcn.get_adjacency_matrix(self.outward, self.num_node)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
