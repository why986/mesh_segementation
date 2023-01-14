import trimesh, os
import numpy as np
from scipy.sparse import csr_matrix
import argparse
from utils import *
from trimesh import exchange

class MeshSolver():
    # load mesh file using trimesh, and calculate the weight matrix for segmentation

    def __init__(self, args):
        # load file using trimesh
        self.base_name = args.input_file.split('/')[-1]
        self.save_path = args.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.mesh = trimesh.load_mesh(args.input_file) 
        self.face_list = self.mesh.faces
        self.num_faces = self.face_list.shape[0]
        self.face_normals = self.mesh.face_normals
        self.vertices = self.mesh.vertices
        self.face_adjacency = self.mesh.face_adjacency
        self.face_adjacency_convex = self.mesh.face_adjacency_convex
        self.face_adjacency_edges = self.mesh.face_adjacency_edges
        self.face_adjacency_unshared = self.mesh.face_adjacency_unshared

        self.ang_eta = args.ang_eta
        self.weight_delta = args.weight_delta
        self.stop_eps = args.stop_eps
        self.MAX_NUM_SEEDS = 10

        self.color_cnt = 0
        self.color = {}
    
    def calc_geod_dist(self, idx):
        # calculate the geodesic distance between two faces
        v_shared = self.vertices[self.face_adjacency_edges[idx]]
        v_unshared = self.vertices[self.face_adjacency_unshared[idx]]
        # relative position of the vertices, treat v_shared[1] as the origin
        x_shared = np.array(v_shared) - np.array(v_shared[1])
        x_unshared = np.array(v_unshared) - np.array(v_shared[1])
        # calculate the rotation matrix
        n0, n1 = self.face_normals[self.face_adjacency[idx]]
        n0 = n0 / np.linalg.norm(n0)
        n1 = n1 / np.linalg.norm(n1)
        n0_n1 = np.dot(n0, n1)
        n0_n1 = np.clip(n0_n1, -1, 1)
        angle = np.arccos(n0_n1) if self.face_adjacency_convex[idx] else -np.arccos(n0_n1)
        x0, y0, z0 = x_shared[0]
        xn, yn, zn = x_shared[0] / np.linalg.norm(x_shared[0])
        M = x_shared[0].dot(x_shared[0] / np.linalg.norm(x_shared[0]))
        c = np.cos(angle)
        s = np.sin(angle)
        rotate_matrix = np.array(
            [
                [xn * xn * (1 - c) + c, xn * yn * (1 - c) - zn * s, xn * zn * (1 - c) + yn * s, (x0 - xn * M) * (1 - c) + (zn * y0 - yn * z0) * s],
                [xn * yn * (1 - c) + zn * s, yn * yn * (1 - c) + c, yn * zn * (1 - c) - xn * s, (y0 - yn * M) * (1 - c) + (xn * z0 - zn * x0) * s],
                [xn * zn * (1 - c) - yn * s, yn * zn * (1 - c) + xn * s, zn * zn * (1 - c) + c, (z0 - zn * M) * (1 - c) + (yn * x0 - xn * y0) * s],
                [0, 0, 0, 1]
            ]
        )
        # calculate the geodesic distance
        center0 = np.mean(self.vertices[self.face_list[self.face_adjacency[idx][0]]], axis=0) - v_shared[1] # center of the triangle
        center1 = np.mean(self.vertices[self.face_list[self.face_adjacency[idx][1]]], axis=0) - v_shared[1]
        if np.dot(np.cross(x_shared[0], x_unshared[0]), n0) > 0:
            rotate_center = rotate_matrix.dot(np.array([*center0, 1.0]))[:3]
            return np.linalg.norm(rotate_center - center1)
        else:
            rotate_center = rotate_matrix.dot(np.array([*center1, 1.0]))[:3]
            return np.linalg.norm(rotate_center - center0)
    
    def calc_ang_dist(self, idx):
        # calculate the angular distance between two faces
        u, v = self.face_adjacency[idx]
        n1 = self.face_normals[u]
        n2 = self.face_normals[v]
        if self.face_adjacency_convex[idx]:
            return self.ang_eta * (1.0 - np.dot(n1, n2))
        else:
            return 1 - np.dot(n1, n2)
    
    def calc_weight_matrix(self):
        geod_dists = []
        ang_dists = []
        for idx in range(len(self.face_adjacency)):
            ang_dist = self.calc_ang_dist(idx)
            ang_dists.append(ang_dist)
            geod_dist = self.calc_geod_dist(idx)
            geod_dists.append(geod_dist)
        ang_dists = np.array(ang_dists)
        geod_dists = np.array(geod_dists)
        
        rows = [pair[0] for pair in self.face_adjacency]
        cols = [pair[1] for pair in self.face_adjacency]
        weight = self.weight_delta * geod_dists / geod_dists.mean() + (1.0 - self.weight_delta) * ang_dists / ang_dists.mean()
        
        weight_mat = csr_matrix((weight, (rows, cols)), shape=(self.num_faces, self.num_faces))
        ang_dist_mat = csr_matrix((ang_dists, (rows, cols)), shape=(self.num_faces, self.num_faces))        
        weight_mat = weight_mat + weight_mat.getH()
        ang_dist_mat = ang_dist_mat + ang_dist_mat.getH()

        return weight_mat, ang_dist_mat
    
    def test(self):
        geod_dist = self.calc_geod_dist(0)
        print(geod_dist)
    
    def visualize(self):
        # colors: Firebrick1, SkyBlue, LawnGreen, Yellow, Purple, DarkOrange1, Red3, Pink1, NavyBlue, Grey
        color_dic = [(255, 48, 48, 200), (135, 206, 235, 200), (124, 252, 0, 200), (255, 255, 0, 200), (160, 32, 240, 200),
                     (255, 127, 0, 200), (205, 0, 0, 200), (255, 181, 197, 200), (0, 0, 128, 200), (128, 128, 128, 200)]
        for i in range(self.num_faces):
            if self.color[i] >= 0:
                self.mesh.visual.face_colors[i] = color_dic[self.color[i]%10]
            else:
                print("Face %d is not colored. (%d)." % (i, self.color[i]))
                self.mesh.visual.face_colors[i] = (0, 0, 0, 200)  # (124, 252, 0, 100)
        self.mesh.show()
        result = exchange.ply.export_ply(self.mesh)
        output_file = open(os.path.join(self.save_path, self.base_name), "wb")
        output_file.write(result)
        output_file.close()

    def select_seeds(self, dis, faces):
        # dis: distance matrix
        # vertices: vertices of the sub mesh
        # return: the selected seeds
        # choose the first seed which is the nearest of other faces, i.e. the sum of distance to other faces is the smallest
        first_seed = -1
        min_sum = float("inf")
        for i in faces:
            sum_dis = 0.
            for j in faces:
                if dis[i][j] != float("inf"):
                    sum_dis += dis[i][j]
            if sum_dis < min_sum and sum_dis != 0:
                min_sum = sum_dis
                first_seed = i
        # print('first seed', first_seed, min_sum)
        # choose seeds which are the farthest from former seeds
        seeds = [first_seed]
        G = [0]
        
        for i in range(self.MAX_NUM_SEEDS-1):
            maxmin_dist = 0
            maxmin_idx = -1
            for j in faces:
                if j not in seeds:
                    min_dist = min([dis[j][seed] for seed in seeds])
                    if min_dist > maxmin_dist and min_dist != float("inf"):
                        maxmin_dist = min_dist
                        maxmin_idx = j
            seeds.append(maxmin_idx)
            G.append(maxmin_dist)
        # automatically determine the number of seeds
        ma = 0
        # print(G, seeds)
        for i in range(self.MAX_NUM_SEEDS-2):
            if G[i] - G[i+1] > ma:
                ma = G[i] - G[i+1]
                num_seeds = i+1
        return seeds[:num_seeds]
    
    def k_way_segmentation(self, seeds, dis, faces):
        for seed in seeds:
            self.color[seed] = self.color_cnt
            self.color_cnt += 1

        fuzzy_faces = defaultdict(list)
        for i in faces:
            sum_prob, max_prob, sub_max_prob = 0., 0., 0.
            max_seed, sub_max_seed = -1, -1
            for seed in seeds:
                if i == seed:
                    prob = 1.
                elif dis[i][seed] == 0:
                    prob = 0.
                else:
                    prob = 1 / dis[i][seed]
                sum_prob += prob
                if prob > max_prob:
                    sub_max_prob = max_prob
                    sub_max_seed = max_seed
                    max_prob = prob
                    max_seed = seed
                elif prob > sub_max_prob:
                    sub_max_prob = prob
                    sub_max_seed = seed
            if (max_prob - sub_max_prob) / sum_prob > 0.01:
                self.color[i] = self.color[max_seed]
            else:
                # if the max prob is not much larger than the second max prob, then the face belongs to a fuzzy part
                self.color[i] = -1
                fuzzy_faces[(max_seed, sub_max_seed)].append(i)
                fuzzy_faces[(sub_max_seed, max_seed)].append(i)
        return fuzzy_faces
    

    def elimate_fuzzy_parts(self, seeds, ang_dist_mat, fuzzy_faces, faces):
        # seeds: the seeds of the segmentation
        # ang_dist_mat: the angular distance matrix
        # fuzzy_faces: the faces that belong to fuzzy parts
        # faces: faces of the sub mesh
        # return: the segmentation result after elimate fuzzy parts
        for i in range(len(seeds)):
            for j in range(i+1, len(seeds)):
                if not (seeds[i], seeds[j]) in fuzzy_faces or len(fuzzy_faces[(seeds[i], seeds[j])]) == 0:
                    continue
                
                flow_graph = build_flow_graph(ang_dist_mat, fuzzy_faces[(seeds[i], seeds[j])], seeds[i], seeds[j], self.color, faces)
                flow_graph.calc_max_flow()
                belongs_to_S1, belongs_to_S2 = flow_graph.get_result()
                # print('S1(%d): ' % seeds[i], belongs_to_S1, 'S2(%d): ' % seeds[j], belongs_to_S2, fuzzy_faces[(seeds[i], seeds[j])])
                for u in belongs_to_S1:
                    if u > 0:
                        self.color[u] = self.color[seeds[i]]
                for u in belongs_to_S2:
                    if u > 0:
                        self.color[u] = self.color[seeds[j]]
    
    def build_sub_graph(self, faces, sub_color):
        # faces: the faces of the father mesh
        # sub_color: the color of the sub mesh
        # return: the sub graph of the sub mesh
        faces_sub = set()
        for i in faces:
            if self.color[i] == sub_color:
                faces_sub.add(i)
        return faces_sub
    
    def solve(self, faces, dis, ang_dist_mat):
        seeds = self.select_seeds(dis, faces)
        if check_seeds(seeds, dis, self.stop_eps):
            return
        print('choose %d seeds' % len(seeds), seeds)
        color_begin = self.color_cnt
        fuzzy_faces = self.k_way_segmentation(seeds, dis, faces)
        self.elimate_fuzzy_parts(seeds, ang_dist_mat, fuzzy_faces, faces)
        for i in range(color_begin, self.color_cnt):
            faces_sub = self.build_sub_graph(faces, i)
            self.solve(faces_sub, dis, ang_dist_mat)



if __name__ == '__main__':
    args = argparse.Namespace()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help="mesh file for segmentation")
    parser.add_argument('--ang-eta', type=float, default=0.3, help="eta for angle distance")
    parser.add_argument('--weight-delta', type=float, default=0.5, help="delta for weight")

    args = parser.parse_args()
    mesh_solver = MeshSolver(args)
    mesh_solver.test()