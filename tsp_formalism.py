import numpy as np
import networkx as nx
import random

N = 5

A = np.zeros((N, N), dtype=float)
for i in np.arange(N):
    for j in np.arange(N):
        if i == j:
            A[i][j] = 0
        elif i > j:
            A[i][j] = A[j][i]
        else:
            A[i][j] = random.uniform(0, 1)

print(A)

D = np.zeros((N**2, N**2), dtype=float)
for i in np.arange(N**2):
    for j in np.arange(N**2):
        if i == j:
            D[i][j] = 0
        elif i > j:
            D[i][j] = D[j][i]
        elif (i%N == (j-1)%N) or ((i-1)%N == j%N):
            D[i][j] = A[i//N][j//N]
print("\n\n")

print(D)

C = np.zeros((N**2, N**2), dtype=int)
for i in np.arange(N**2):
    for j in np.arange(N**2):
        if i == j:
            C[i][j] = -N
        elif ((i//N == j//N) or (i%N == j%N)):
            C[i][j] = N
        else:
            C[i][j] = -1
print("\n\n")

print(C)

print("\n\n")

print(C+D)

X = np.identity(N)
X = np.matrix.flatten(X)

E_D = np.matmul(np.matmul(np.transpose(X), D), X)

print(f"\n\n E_D = {E_D}")

E_D_exp = 0
for i in np.arange(N):
    E_D_exp += A[i][(i+1)%N]
print(f"\n\n E_D_exp = {2*E_D_exp}")

E = np.matmul(np.matmul(np.transpose(X), C+D), X)

print(f"\n\nE = {E}")
