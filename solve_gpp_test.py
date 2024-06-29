#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 14:48:37 2024

@author: sadler

solving all graph types

run using:
"""

import sys
import os
import time
import re

import numpy as np
import networkx as nx
# import metis

from constants_equations import *
from hnn import *
from graph_functions import *

graph_type = 'trivial'
N = 7
time_str = time.strftime("%Y%m%d-%H%M%S")


graph_name = f'gpp_{N}_{graph_type}_{time_str}'
parent_path = os.path.join(save_to, graph_name)
os.makedirs(parent_path, exist_ok=True)

# optional arguments
num_runs = 2



# scale free graphs
if graph_type == 'SF':
    G = nx.scale_free_graph(N, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=r)
    A = nx.to_numpy_array(G)

# Small world graphs
elif graph_type == 'WS':
    k = float(sys.argv[3])
    p = float(sys.argv[4])
    G = nx.watts_strogatz_graph(N, k, p, seed=r)
    A = nx.to_numpy_array(G)

# Random graphs
elif graph_type == 'ER':
    p = float(sys.argv[3])
    G = nx.erdos_renyi_graph(N, p, seed=r)
    A = nx.to_numpy_array(G)

# Trivial graphs
elif graph_type == 'trivial':
   
    A = trivial_graphs[N - 3]
    G = nx.from_numpy_array(A)


# Save information about this graph and the simulation
with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''Graph information for {graph_name}
Simulation started on {time_str} \n''')
    f.write(f'''\nnumber of nodes is: {N}
            
Constants:
    seed: {r}
    Q_alpha: {Q_alpha}
    MM_gamma: {MM_gamma}
    ''')
    f.close()

# create csv files about this graph
os.makedirs(f'{parent_path}/csv', exist_ok=True)
np.savetxt(f'{parent_path}/csv/A.csv', A, delimiter=',')




# Benchmark Partitions
# metis_x_list = metis.part_graph(G, 2)[1]
# metis_x = np.array(metis_x_list, dtype=int) # changing to an np array for ease of use

kernighan_lin_coms = nx.algorithms.community.kernighan_lin.kernighan_lin_bisection(G)
kernighan_lin_x = get_x_from_coms(kernighan_lin_coms)

comp = nx.algorithms.community.girvan_newman(G)
girvan_newman_coms = tuple(sorted(c) for c in next(comp))
if len(girvan_newman_coms) > 2:
    with open(f'{parent_path}/information.txt', 'a') as f:
        f.write(f'Girvan_Newman found more than 2 partitions!!!\nIgnore these results!!!\n')
        f.close()
girvan_newman_x = get_x_from_coms(girvan_newman_coms)

# metis_laypunov_M, metis_laypunov_Q = get_partition_quality(metis_x, A)
kernighan_lin_laypunov_M, kernighan_lin_laypunov_Q = get_partition_quality(kernighan_lin_x, A)
girvan_newman_laypunov_M, girvan_newman_laypunov_Q = get_partition_quality(girvan_newman_x, A)

# plot_partition(G, metis_x, "metis_partition", parent_path)
plot_partition(G, kernighan_lin_x, "kernighan_lin_partition", parent_path)
plot_partition(G, girvan_newman_x, "girvan_newman_partition", parent_path)

# metis & {metis_laypunov_M:.{3}g} & {metis_laypunov_Q:.{3}g} \\\\
#      & \\rowcolor{{yellow}} 
with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''\nBenchmark Partitions:
    Partition Type & laypunov_M & laypunov_Q
    \\multirow{{{num_runs + 6}}}{{*}}{{{N}}} & \\rowcolor{{yellow}} kernighan-lin & {kernighan_lin_laypunov_M:.{3}g} & {kernighan_lin_laypunov_Q:.{3}g} \\\\
     & \\rowcolor{{yellow}} girvan-newman & {girvan_newman_laypunov_M:.{3}g} & {girvan_newman_laypunov_Q:.{3}g} \\\\ \n''')
    f.close()



# Ideal Partitions
ideal_laypunov_M_x, ideal_laypunov_Q_x = get_ideal_partitions(A)

plot_partition(G, ideal_laypunov_M_x, "ideal_laypunov_M_x", parent_path)
plot_partition(G, ideal_laypunov_Q_x, "ideal_laypunov_Q_x", parent_path)

ideal_laypunov_M_laypunov_M, ideal_laypunov_M_laypunov_Q = get_partition_quality(ideal_laypunov_M_x, A)
ideal_laypunov_Q_laypunov_M, ideal_laypunov_Q_laypunov_Q = get_partition_quality(ideal_laypunov_Q_x, A)

with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''\nIdeal Partitions:
Partition Type & laypunov_M & laypunov_Q
     & \\rowcolor{{blue}} ideal_laypunov_M & {ideal_laypunov_M_laypunov_M:.{3}g} & {ideal_laypunov_M_laypunov_Q:.{3}g} \\\\
     & \\rowcolor{{blue}} ideal_laypunov_Q & {ideal_laypunov_Q_laypunov_M:.{3}g} & {ideal_laypunov_Q_laypunov_Q:.{3}g} \\\\ ''')
    f.close()




# SNN Partitions
simulation_time = get_simulation_time(N) #how long the simulation needs to run for

Q = get_GPP_QUBO(A)
M = get_modularity_matrix(A)

b = np.full(N, ic)

for M_type, HNN_M in {'QUBO': Q, 'Modularity': M}.items():
    W = (max_weight/np.max(np.abs(HNN_M)))*HNN_M

    with open(f'{parent_path}/information.txt', 'a') as f:
        f.write(f'''\nSNN {M_type} Partitions:
    simulation time is: {simulation_time} tau
        Partition Type & laypunov_M & laypunov_Q \n''')
        f.close()

    for run in np.arange(num_runs):
        r_snn = random.randint(0, 2**23-1)
        path = f'{parent_path}/{run}_{M_type}'
        os.makedirs(path, exist_ok=True)
        with open(f'{path}/information.txt', 'a') as f:
            f.write(f"Information for run number {run} seed {r_snn}\n")
            f.close()
        
        spikes_array = get_SNN_partition(W, b, simulation_time, path, r=r_snn)

        snn_x = get_x(spikes_array)
        plot_partition(G, snn_x, "partitioned_graph", path)
        snn_laypunov_M, snn_laypunov_Q = get_partition_quality(snn_x, A)
        with open(f'{parent_path}/information.txt', 'a') as f:
            f.write(f'\t & \\rowcolor{{pink}} SNN \#{run} & {snn_laypunov_M:.{3}g} & {snn_laypunov_Q:.{3}g}\\\\ \n')
            f.close()