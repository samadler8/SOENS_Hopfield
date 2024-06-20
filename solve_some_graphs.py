import os
from example_graphs import *
from partition_functions import *


current_directory = os.getcwd()


N = 25
A = WS_adjacency_matrix(N, 0.5, 0.1)
# A = trivial_graphs[3]

Q = get_Q(A)
print(Q)

modularity_x, laypunov_M_x, laypunov_Q_x, QUBO_x = get_ideal_partitions(A, alpha=1, gamma=1)

path_pics = os.path.join(current_directory, 'pics')

plot_partition(modularity_x, A, "Ideal Modularity", my_path=path_pics)
plot_partition(laypunov_M_x, A, "Ideal Laypunov Modularity Matrix", my_path=path_pics)
plot_partition(laypunov_Q_x, A, "Ideal Laypunov Q Matrix", my_path=path_pics)
plot_partition(QUBO_x, A, "Ideal QUBO Original Formulation", my_path=path_pics)
