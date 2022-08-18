from ManifoldLearning import LE
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


#Connecting K nearest neighbors
K = 5
m = 2
LaplacianEigen = LE(s_curve, K, m, adjacency='k-nearest')
Y = LaplacianEigen.DimensionReduction()

#Plot Data
fig = plt.figure(figsize=(20,10))
LaplacianEigen.plot_data(s_curve, color, 121, '3d', 's-curve')
LaplacianEigen.plot_data(Y, color, 122, '2d', 'LE')
LaplacianEigen.plot_graph(s_curve, LaplacianEigen.A, 121, '3d')
plt.show()