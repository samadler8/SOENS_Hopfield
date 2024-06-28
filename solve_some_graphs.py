import os
from example_graphs import *
from partition_functions import *
from gpp_helpers import *
import time

time_str = time.strftime("%Y%m%d-%H%M%S")


current_directory = os.getcwd()

n = 10
k = 0.5
p = 0.1

# G = nx.watts_strogatz_graph(n, k, p, seed=seed)
# G = nx.scale_free_graph(n, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=seed)
G = nx.erdos_renyi_graph(n, p)

A = nx.to_numpy_array(G)

print(A)

modularity_x, laypunov_M_x, laypunov_Q_x, QUBO_x = get_ideal_partitions(A)

path_pics = os.path.join(current_directory, 'pics')

plot_partition(modularity_x, A, f"{time_str}_Ideal Modularity", my_path=path_pics)
plot_partition(laypunov_M_x, A, f"{time_str}_Ideal Laypunov Modularity Matrix", my_path=path_pics)
plot_partition(laypunov_Q_x, A, f"{time_str}_Ideal Laypunov Q Matrix", my_path=path_pics)
plot_partition(QUBO_x, A, f"{time_str}_Ideal QUBO Original Formulation", my_path=path_pics)


plot_partition(np.abs(modularity_x - laypunov_M_x), A, f"{time_str}_modularity_difference", my_path=path_pics)
plot_partition(np.abs(QUBO_x - laypunov_Q_x), A, f"{time_str}_QUBO_difference", my_path=path_pics)

plot_partition(np.abs(laypunov_M_x - laypunov_Q_x), A, f"{time_str}_check", my_path=path_pics)