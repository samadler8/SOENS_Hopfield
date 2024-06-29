"""
Figuring out Constraint matrix through trial and error
"""

import numpy as np


N = 100 #number of nodes

different_row_column_value = 0 #1
same_row_column_value = -1 #-N
# if not(same_row_column_value < -N*different_row_column_value/2):
#     print("Invalid parameters for same/different row/column weights")


# diagonal_value = -(N-1)*(2*same_row_column_value + (N-1)*different_row_column_value)
diagonal_value = 0 #N

C = np.zeros((N**2, N**2), dtype=int)
for i in np.arange(N**2):
    for j in np.arange(N**2):
        if i == j:
            C[i][j] = diagonal_value
        elif ((i//N == j//N) or (i%N == j%N)):
            C[i][j] = same_row_column_value
        else:
            C[i][j] = different_row_column_value

print(C)

X_1 = np.ones(N*N, dtype=int)
X_1p = np.copy(X_1)
X_1p[0] = 0
X_0 = np.zeros(N*N, dtype=int)
X_0p = np.copy(X_0)
X_0p[0] = 1
X_I = np.matrix.flatten(np.identity(N, dtype=int))
X_Ip = np.copy(X_I)
X_Ip[1] = 1
X_Is = np.copy(X_I)
X_Is[0:N] = X_I[N:2*N]
X_Is[N:2*N] = X_I[0:N]
X_W = np.zeros((N, N), dtype=int)
X_W[0][0:N-1] = np.ones(N-1, dtype=int)
X_W[1][N-1] = 1
X_WT = np.transpose(np.copy(X_W))
X_W = np.matrix.flatten(X_W)
X_WT = np.matrix.flatten(X_WT)
X_B = np.zeros(N*N, dtype=int)
X_B[0:N] = np.ones(N, dtype=int)
Xs = [X_1, X_0, X_1p, X_0p, X_B, X_W, X_WT, X_Ip, X_I, X_Is]
for X in Xs:
    print(X)
    #print(np.matmul(np.transpose(X), C))
    E = np.matmul(np.matmul(np.transpose(X), C), X)
    print(E)
    print("\n\n")
