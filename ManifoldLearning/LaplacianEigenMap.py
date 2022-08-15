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
K = 4
A = LaplacianEigen.Adjacency(K, method='k-nearest')

#Evaluate Graph Laplacian
L = LaplacianEigen.Laplacian(A)
