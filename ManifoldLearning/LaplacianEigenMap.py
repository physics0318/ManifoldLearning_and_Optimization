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

LaplacianEigen = ManifoldLearning(s_curve)

#Connecting K nearest neighbors
K = 5
A = LaplacianEigen.Adjacency(K, method='k-nearest')

#Evaluate Graph Laplacian
L = LaplacianEigen.Laplacian(A)

eigenValue, eigenFunction = np.linalg.eigh(L)


Y = eigenFunction[:,:2]

#Plot Data
fig = plt.figure(figsize=(20,10))
LaplacianEigen.plot_data(s_curve, color, 121, '3d')
LaplacianEigen.plot_data(Y, color, 122, '2d')
LaplacianEigen.plot_graph(s_curve, A, 121, '3d')
plt.show()