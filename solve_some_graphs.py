import os
from example_graphs import *
from partition_functions import *
from gpp_helpers import *

seed=None

current_directory = os.getcwd()

n = 1000
k = 0.5
p = 0.1

alpha = 1
gamma = 1

G = nx.watts_strogatz_graph(n, k, p, seed=seed)
G = nx.scale_free_graph(n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=seed)
G = nx.erdos_renyi_graph(n, p, seed=seed)

A = nx.adjacency_matrix(G)

modularity_x, laypunov_M_x, laypunov_Q_x, QUBO_x = get_ideal_partitions(A, alpha=alpha, gamma=gamma)

path_pics = os.path.join(current_directory, 'pics')

plot_partition(modularity_x, A, "Ideal Modularity", my_path=path_pics)
plot_partition(laypunov_M_x, A, "Ideal Laypunov Modularity Matrix", my_path=path_pics)
plot_partition(laypunov_Q_x, A, "Ideal Laypunov Q Matrix", my_path=path_pics)
plot_partition(QUBO_x, A, "Ideal QUBO Original Formulation", my_path=path_pics)
