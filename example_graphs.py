#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 17:42:10 2023

@author: sadler

Functions for generating adjacency matricies for graphs
"""

import numpy as np
import random
import math

graph_types = ['trivial', 'WS', 'WS_SF', 'ER', 'SF']

# Scale-free graph generator
def SF_adjacency_matrix(N, C, G_SF):
    C = C*N
    M = np.empty((N,N), dtype=int)
    i = 0
    for i in np.arange(N):
        P = random.uniform(0, 0.66)
        K = round((C/P)**(1/G_SF))
        P = K/N
        for j in np.arange(N):
            if j < i:
                M[i][j] = M[j][i]
            elif i == j:
                M[i][j] = 0
            else:
                r = random.uniform(0, 1)
                if r > P:
                    M[i][j] = 0
                else:
                    M[i][j] = 1
    return M

# Small-world scale-free graph generator
def WS_SF_adjacency_matrix(N, C, B, G_SF):
    C = C*N
    M = np.zeros((N,N), dtype=int)
    i = 0
    while i < N:
        k = 1
        P = random.uniform(0, 0.66)
        K = round((C/P)**(1/G_SF))
        if K > N:
            K = N - 1
        while k <= K/2:
            # usually connect right side
            r = random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
            k += 1
        # say K = 3 and beta = 0, half the time it will have two left connections and half the time it will only have one
        r = random.uniform(0, 1)
        if r < (K/2 + 1 - k):
            # usually connect right side
            r = random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
        i += 1  
    return M

# Random graph
def ER_adjacency_matrix(N, density):
    M = np.empty((N,N), dtype=int)
    for i in np.arange(N):
        for j in np.arange(N):
            if j < i:
                M[i][j] = M[j][i]
            elif i == j:
                M[i][j] = 0
            else:
                r = random.uniform(0, 1)
                if r > density:
                    M[i][j] = 0
                else:
                    M[i][j] = 1
    return M

# Small-world graph generator
def WS_adjacency_matrix(N, K, B):
    K = K*N
    M = np.zeros((N,N), dtype=int)
    i = 0
    while i < N:
        k = 1
        while k <= K/2:
            # usually connect right side
            r = random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
            k += 1
        # say K = 3 and beta = 0, half the time it will have two left connections and half the time it will only have one
        r = random.uniform(0, 1)
        if r < (K/2 + 1 - k):
            # usually connect right side
            r = random.uniform(0, 1)
            if r > B:
                if (i + k) >= N:
                    M[i][i + k - N] = 1
                    M[i + k - N][i] = 1
                else:
                    M[i][i + k] = 1
                    M[i + k][i] = 1         
            # random connections
            else:
                n = random.randint(0, math.ceil((N - K - 1)))
                n += i + round(K/2)
                if n >= N:
                    n -= N
                M[i][n] = 1
                M[n][i] = 1
        i += 1  
    return M

# trivial graphs
three_node = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
four_node = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
five_node = np.array([[0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 0]])
six_node = np.array([[0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
seven_node = np.array([[0, 1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1], [0, 0, 0, 0, 0, 1, 0]])
trivial_graphs = [three_node, four_node, five_node, six_node, seven_node]

