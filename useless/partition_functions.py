import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
import matplotlib.pyplot as plt
import os

Q_alpha = 1
MM_gamma = 1

def get_GPP_QUBO(A):
    '''
    returns QUBO matrix for GPP
    '''
    N = A.shape[0]
    Q = A - Q_alpha
    for i in np.arange(N):
        Q[i][i] = Q_alpha*(N - 1) - np.sum(A[i])
    return Q

def get_modularity_matrix(A):
    '''
    returns the modularity matrix
    '''
    m2 = np.sum(A) # 2 times number of edges
    g = np.sum(A, axis=1) # degree matrix
    M = A - (MM_gamma/m2)*np.outer(g, g)
    print(M)
    return M

def get_coms_from_x(x):
    coms = [[], []]
    for i in np.arange(x.size):
        if x[i] == 0:
            coms[0].append(i)
        else:
            coms[1].append(i)
    return coms

def get_s_from_x(x):
    s = 2*x - np.ones((x.size))
    return s

def get_modularity(x, A):
    '''
    Ideal when max
    '''
    coms = get_coms_from_x(x)
    G = nx.from_numpy_array(A)
    modularity = nx.algorithms.community.quality.modularity(G, coms, resolution=1)
    return modularity

def get_laypunov(x, W, b=0):
    '''
    This is the Laypunov energy of our SNN
    Should be minimized
    '''
    if b == 0:
        b = np.zeros((x.size))
    V = (-1/2)*np.matmul(np.matmul(np.transpose(x), W), x) - np.matmul(np.transpose(b), x)
    return V

def get_original_QUBO(x, A, alpha=1):
    '''
    Got this equation from minimizing cuts (s^{T} -A s)
    and minimizing constraint (s^{T} 1_{n x n} s)
    Ideal when E is minimum
    '''
    s = get_s_from_x(x)
    E = np.matmul(np.matmul(np.transpose(s), (alpha*np.ones((A.shape)) - A)), s)
    return E

def get_ideal_partitions(A):
    N = A.shape[0]
    Q = get_GPP_QUBO(A)
    M = get_modularity_matrix(A)
    x = np.zeros(N, dtype = int)

    modularity = get_modularity(x, A)
    laypunov_M = get_laypunov(x, M)
    laypunov_Q = get_laypunov(x, Q)
    QUBO = get_original_QUBO(x, A)

    modularity_x = x.copy()
    laypunov_M_x = x.copy()
    laypunov_Q_x = x.copy()
    QUBO_x = x.copy()

    while not np.all(x == 1):
        x[-1] += 1
        i = N
        while i>0:
            i -= 1
            if x[i] == 2:
                x[i] = 1
                x[i-1] += 1
        
        modularity_temp = get_modularity(x, A)
        laypunov_M_temp = get_laypunov(x, M)
        laypunov_Q_temp = get_laypunov(x, Q)
        QUBO_temp = get_original_QUBO(x, A)

        if modularity_temp > modularity:
            modularity = modularity_temp
            modularity_x = x.copy()
        if laypunov_M_temp < laypunov_M:
            laypunov_M = laypunov_M_temp
            laypunov_M_x = x.copy()
        if laypunov_Q_temp < laypunov_Q:
            laypunov_Q = laypunov_Q_temp
            laypunov_Q_x = x.copy()
        if QUBO_temp < QUBO:
            QUBO = QUBO_temp
            QUBO_x = x.copy()
    return modularity_x, laypunov_M_x, laypunov_Q_x, QUBO_x

def plot_partition(x, A, title, my_path=''):
    N = x.size
    G = nx.from_numpy_array(A)

    color_map = np.empty(N, dtype=str)
    for i in np.arange(N):
        if x[i] == 1:
            color_map[i] = 'red'
        else:
            color_map[i] = 'cyan'
    
    plt.clf()
    plt.close()
    plt.figure()
    nx.draw(G, node_color=color_map, with_labels=True)
    os.makedirs(f'{my_path}', exist_ok=True)
    plt.savefig(f'{my_path}/{title}.png')
    plt.clf()
    plt.close()
    return