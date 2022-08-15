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

#Connecting K nearest neighbors
K = 4
A = Isomap.Adjacency(K, method='k-nearest')

#Create Geodesic Distance Matrix
G = [[0,1,1,3,3,7],
    [1,0,1,2,3,7],
    [1,1,0,3,2,6],
    [3,2,3,0,3,7],
    [3,3,2,3,0,4],
    [7,7,6,7,4,0]]

I = np.identity(6)
J = np.ones((6,6))
H = I-(1/6)*J
D = np.square(G)
B = -(1/2)*H@D@H

EigenValues,  EigenFunctions = np.linalg.eig(B)

L = np.diag(EigenValues[:2])
L = np.sqrt(L)
E = EigenFunctions[:,:2]
print(EigenFunctions)
print(E)

X = np.matmul(E,L)

ax = plt.subplot(111)
ax.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Spectral)
for i in range(len(X)):
    plt.text(X[i][0], X[i][1], str(i))
plt.show()
