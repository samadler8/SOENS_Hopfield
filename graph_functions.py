#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Jun 20 16:12:12 2024

@author: sadler

Partition & Graph functions
"""

import numpy as np
import math
import random
from constants_equations import *



def get_random_A(N):
    A = np.zeros((N, N), dtype=float)
    for i in np.arange(N):
        for j in np.arange(N):
            if i == j:
                A[i][j] = 0
            elif i > j:
                A[i][j] = A[j][i]
            else:
                A[i][j] = random.uniform(0, 1)
    return A


def get_TSP_QUBO(A): # get the Q matrix
    N = A.shape[0]
    N2 = A.size
    max_A = np.max(A)
    Q = np.zeros((N2, N2), dtype=int)
    for i in np.arange(N2):
        for j in np.arange(N2):
            if i > j: # graph is symmetric
                Q[i][j] = Q[j][i]
            elif (i%N == (j-1)%N) or ((i-1)%N == j%N): # excite by inverse of distance so minimum distance is preferred
                Q[i][j] = max_A/(A[i//N][j//N])
            elif ((i//N == j//N) or (i%N == j%N)): # inhibit neurons in same position or representing same city
                Q[i][j] = -1
    Q = (max_weight/np.max(Q))*Q
    return Q

def get_GPP_QUBO(A): # get the Q matrix
    N = A.shape[0]
    Q = A - Q_alpha
    for i in np.arange(N):
        Q[i][i] = Q_alpha*(N - 1) - np.sum(A[i])
    Q = (max_weight/np.max(Q))*Q
    return Q

def get_modularity_matrix(A): # Get the modulatity matrix
    m2 = np.sum(A) # 2 times number of edges
    g = np.sum(A, axis=1) # degree matrix
    M = A - (MM_gamma/m2)*np.outer(g, g)
    M = (max_weight/np.max(M))*M
    return M

def get_x(spikes_array):
    return (spikes_array > 0.25).astype(int)

def get_route(x):
    N = math.sqrt(x.size)

    route_matrix = x.reshape(N, N)

    route = []
    for i in range(N):
        start_node = i
        end_node = np.where(route_matrix[i] == 1)[0][0]
        route.append((start_node, end_node))

    return route

def get_cycle(x):
    N = math.sqrt(x.size)

    route_matrix = x.reshape(N, N)

    cycle = np.zeros(N+1, dtype=int)
    for i in range(N):
        cycle[i] = np.where(route_matrix[i] == 1)[0][0]
    cycle[-1] = cycle[0]
    return cycle

def get_distance(cycle, A):
    distance_traveled = 0
    for i in np.arange(len(cycle)-1):
        distance_traveled += A[cycle[i]][cycle[i+1]]
    return distance_traveled


def get_x_from_coms(coms):
    length = 0
    for c in coms:
        length += len(c)
    x = np.zeros(length, dtype=int)
    for i in np.arange(length):
        if i in coms[0]:
            x[i] = 1
    return np.array(x)


def get_laypunov(x, W, b=0):
    '''
    This is the Laypunov energy of our SNN
    Should be minimized
    '''
    if b == 0:
        b = np.zeros((x.size))
    V = (-1/2)*np.matmul(np.matmul(np.transpose(x), W), x) - np.matmul(np.transpose(b), x)
    return V

def get_partition_quality(x, A):
    Q = get_GPP_QUBO(A)
    M = get_modularity_matrix(A)

    laypunov_M = get_laypunov(x, M)
    laypunov_Q = get_laypunov(x, Q)

    return laypunov_M, laypunov_Q

def get_ideal_partitions(A):
    N = A.shape[0]
    x = np.zeros(N, dtype = int)

    laypunov_M, laypunov_Q = get_partition_quality(x, A)

    laypunov_M_x = x.copy()
    laypunov_Q_x = x.copy()

    while not np.all(x == 1):
        x[-1] += 1
        i = N
        while i>0:
            i -= 1
            if x[i] == 2:
                x[i] = 1
                x[i-1] += 1

        laypunov_M_temp, laypunov_Q_temp = get_partition_quality(x, A)

        if laypunov_M_temp < laypunov_M:
            laypunov_M = laypunov_M_temp
            laypunov_M_x = x.copy()
        if laypunov_Q_temp < laypunov_Q:
            laypunov_Q = laypunov_Q_temp
            laypunov_Q_x = x.copy()
        
    return laypunov_M_x, laypunov_Q_x
