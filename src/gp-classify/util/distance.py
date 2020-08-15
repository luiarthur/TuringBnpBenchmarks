import numpy as np

def squared_euclidean(X, Y):
    return np.power(X[:, None, :] - Y[None, :, :], 2).sum(-1)

def euclidean(X, Y):
    return np.sqrt(squared_euclidean(X, Y))

# TEST.
# def slow(X, Y):
#     M = X.shape[0]
#     N = Y.shape[0]
#     out = np.zeros((M, N))
#     for m in range(M):
#         for n in range(N):
#             out[m, n] = np.power(X[m, :] - Y[n, :], 2).sum()
#     return out
# 
# x = np.random.randn(20, 2)
# y = np.random.randn(15, 2)
# assert np.all(squared_euclidean(x, y) - slow(x, y) == 0)
#
# from scipy.spatial import distance_matrix
# assert np.all(euclidean(x, y) - distance_matrix(x, y) == 0)
# euclidean(x, y) - distance_matrix(x, y)
