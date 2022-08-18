from ManifoldLearning import LLE
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

K = 6
m=2
LLE = LLE(s_curve, K, m, adjacency='k-nearest')
X = LLE.DimensionReduction()

#Plot Data
fig = plt.figure(figsize=(20,10))
LLE.plot_data(s_curve, color, 121, '3d', 's-curve')
LLE.plot_data(X, color, 122, '2d', 'LLE')
LLE.plot_graph(s_curve, LLE.A, 121, '3d')
plt.show()