import time
import argparse
from mesh_solver import MeshSolver
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', type=str, help="mesh file for segmentation")
parser.add_argument('--save-path', type=str, default='results', help="path to save the result")
parser.add_argument('--ang-eta', type=float, default=0.3, help="eta for angle distance")
parser.add_argument('--weight-delta', type=float, default=0.6, help="delta for weight")
parser.add_argument('--stop-eps', type=float, default=24, help="stop eps for recursive segmentation")

args = parser.parse_args()

if __name__ == '__main__':
    mesh_solver = MeshSolver(args)
    start_time = time.time()
    weight_mat, ang_dist_mat = mesh_solver.calc_weight_matrix()
    dis = calc_shortest_dist(mesh_solver.num_faces, weight_mat)
    end_time = time.time()
    print('Calculate shortest distance matrix in {}s.'.format(end_time - start_time))
    mesh_solver.solve(range(mesh_solver.num_faces), dis, ang_dist_mat)
    mesh_solver.visualize()

