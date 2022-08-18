from ManifoldLearning import ManifoldLearning
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import random_projection

#S curve data
N=200       #number of data

s_curve, color = datasets.make_s_curve(N, random_state=0)
digits, labels = datasets.load_digits(return_X_y=True)
rand_proj = random_projection.SparseRandomProjection(n_components=3, random_state=0)
projected_digits = rand_proj.fit_transform(digits)

Isomap = ManifoldLearning(s_curve)
LaplacianEigen = ManifoldLearning(s_curve)
LLE = ManifoldLearning(s_curve)
m = 2

#Connecting K nearest neighbors
K = 4
A = Isomap.Adjacency(K, method='k-nearest')

#Isomap
G = Isomap.FloydWarshall(A)
X = Isomap.MDS(G, m)

#Laplacian Eigenmap
L = LaplacianEigen.Laplacian(A)
eigenValue, eigenFunction = np.linalg.eigh(L)
Y = eigenFunction[:,:m]

#Locally Linear Embedding
G = LLE.GramMatrix(A, K)
w = LLE.WeightsByLagrangeMult(G, K)
Z = LLE.DimReduct(A, w, m)

#Plot Data
fig = plt.figure(figsize=(20,10))
LaplacianEigen.plot_data(s_curve, color, 221, '3d', 's-curve')
LaplacianEigen.plot_graph(s_curve, A, 221, '3d')
LaplacianEigen.plot_data(X, color, 222, '2d', 'Isomap')
LaplacianEigen.plot_data(Y, color, 223, '2d', 'LE')
LaplacianEigen.plot_data(Z, color, 224, '2d', 'LLE')
plt.show()