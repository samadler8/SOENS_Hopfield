#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 21 14:48:37 2024

@author: sadler

solving the traveling salesman problem

run using:
python3 adler/ASC/code/solve_graph.py 
"""

import sys
import os
import time

import numpy as np
import networkx as nx

from constants_equations import *
from hnn import *
from graph_functions import *

r = random.randint(0, 2**23-1)

if len(sys.argv) < 3:
    print("Insufficient parameters")
    sys.exit()

graph_type = sys.argv[1] # graph type (trivial, WS, WS_SF, ER, SF)
N = int(sys.argv[2]) # number of nodes
time_str = time.strftime("%Y%m%d-%H%M%S")

# check for valid graph type
if graph_type not in graph_types:
    print("Invalid graph type has been selected")
    sys.exit()

graph_name = f'tsp_{sys.argv}_{time_str}'
parent_path = os.path.join(save_to, f'{graph_name}')
os.makedirs(parent_path, exist_ok=True)

# optional arguments
num_runs = 1
extra_args = 0
for a in sys.argv:
    if a[:5] == 'runs=':
        if a[5:].isnumeric:
            num_runs = int(a[5:])
        else:
            with open(f'{parent_path}/information.txt', 'a') as f:
                f.write("typo in runs \n")
                f.close()
            sys.exit()
        if num_runs < 1:
            with open(f'{parent_path}/information.txt', 'a') as f:
                f.write("need atleast 1 run1 \n")
                f.close()
            sys.exit()
        extra_args += 1

# general exception
if N < 2:
    with open(f'{parent_path}/information.txt', 'a') as f:
        f.write("the number of nodes is less than 4. This graph is meaningless \n")
        f.close()
    sys.exit()

if graph_type == 'trivial':
    if len(sys.argv) != 3  + extra_args:
        with open(f'{parent_path}/information.txt', 'a') as f:
            f.write("Insufficient parameters \n")
            f.close()
        sys.exit()
    
    # exceptions
    if (N < 3) or (N > 7):
        with open(f'{parent_path}/information.txt', 'a') as f:
            f.write("the number of nodes is outside what I have trivial graphs for \n")
            f.close()
        sys.exit()

    A = trivial_graphs[N - 3]
    

elif graph_type == 'random':
    A = nx.get_random_A(N)
    G = nx.from_numpy_array(A)


# Save information about this graph and the simulation
with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''Graph information for {graph_name}
Simulation started on {time_str} \n''')
    f.write('Arguments:\npython3 ')
    for ar in sys.argv:
        f.write(f'{ar} ')
    f.write(f'''\nnumber of nodes is: {N}
            
Constants:
    seed: {r}
    ''')
    f.close()

# create csv files about this graph
os.makedirs(f'{parent_path}/csv', exist_ok=True)
np.savetxt(f'{parent_path}/csv/A.csv', A, delimiter=',')




# Benchmark Partitions
greedy_cycle = nx.algorithms.approximation.traveling_salesman.greedy_tsp(G, weight="weight", source=None)
christofides_cycle = nx.algorithms.approximation.traveling_salesman.christofides(G, weight="weight", tree=None)
annealing_cycle = nx.algorithms.approximation.traveling_salesman.simulated_annealing_tsp(G, "greedy", weight="weight", source=None, temp=100, move="1-1", max_iterations=10, N_inner=100, alpha=0.01, seed=None)

greedy_distance = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(greedy_cycle))
christofides_distance = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(christofides_cycle))
annealing_distance = sum(G[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(annealing_cycle))

plot_route(G, list(nx.utils.pairwise(greedy_cycle)), f"Greedy - Distance = {greedy_distance}", path=parent_path)
plot_route(G, list(nx.utils.pairwise(christofides_cycle)), f"Greedy - Distance = {christofides_distance}", path=parent_path)
plot_route(G, list(nx.utils.pairwise(annealing_cycle)), f"Greedy - Distance = {annealing_distance}", path=parent_path)

with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''\nBenchmark Partitions:
    Solver & Distance Traveled
    \\multirow{{{num_runs + 6}}}{{*}}{{{N}}} & \\rowcolor{{yellow}} Greedy & {greedy_distance:.{3}g} \\\\
     & \\rowcolor{{yellow}} Christofides & {christofides_distance:.{3}g} \\\\
     & \\rowcolor{{yellow}} Annealing & {annealing_distance:.{3}g} \n''')
    f.close()




# SNN Partitions
simulation_time = get_simulation_time(N) #how long the simulation needs to run for

Q = get_TSP_QUBO(A)
W = (max_weight/np.max(np.abs(Q)))*Q

b = np.full(N, 0.97*ic)


with open(f'{parent_path}/information.txt', 'a') as f:
    f.write(f'''\nSNN Partitions:
simulation time is: {simulation_time} tau
    Partition Type & modularity & laypunov_M & laypunov_Q & QUBO\n''')
    f.close()

for run in np.arange(num_runs):
    r_snn = random.randint(0, 2**23-1)
    path = f'{parent_path}/{run}'
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/information.txt', 'a') as f:
        f.write(f"Information for run number {run} seed {r_snn}\n")
        f.close()
    
    fluxon_array = get_SNN_partition(W, b, simulation_time, path, r=r_snn)

    snn_x = get_x(fluxon_array)
    route = get_cycle(snn_x)
    plot_route(G, route, "Salesman_Route", path)
    snn_distance = get_distance(snn_x, A)
    with open(f'{parent_path}/information.txt', 'a') as f:
        f.write(f'\t & \\rowcolor{{pink}} SNN \#{run} & {snn_distance:.{3}g}\\\\ \n')
        f.close()