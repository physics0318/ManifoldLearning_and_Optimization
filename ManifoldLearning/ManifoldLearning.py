import numpy as np
import matplotlib.pyplot as plt

class ManifoldLearning:
    def __init__(self, data):
        self.data = data
        self.N = len(data)

    def Euclidean(self, x, y):
        if len(x) == len(y):
            d = 0
            for i in range(len(x)):
                d += ((x[i]-y[i])**2)
            d = np.sqrt(d)
        return d

    def kSmallest(self, l, k):
        L = [i for i in l]
        Q = [0 for i in range(len(L))]
        while np.count_nonzero(Q) < k:
            m = L[0]
            ind = 0
            for i in range(len(L)):
                if L[i] < m:
                    m = L[i]
                    ind = i
            L[ind] = float('inf')
            Q[ind] = m
        return Q

    def Adjacency(self, const, method='k-nearest'):
        A = np.zeros((self.N,self.N))
        if method == 'k-nearest':
            for i in range(self.N):
                L = np.zeros(self.N)
                L[i] = float('inf')
                for j in range(self.N):
                    if i != j:
                        L[j] = self.Euclidean(self.data[i], self.data[j])
                Q = self.kSmallest(L, const)
                for k in range(self.N):
                    if Q[k] == 0:
                        A[i][k] = 0
                    else:
                        A[i][k] = 1
            for i in range(self.N):
                for j in range(i, self.N):
                    a = max(A[i][j], A[j][i])
                    A[i][j] = a
                    A[j][i] = a

        elif method == 'epsilon-neighborhood':
            for i in range(self.N):
                for j in range(self.N):
                    d = self.Euclidean(self.data[i], self.data[j])
                    if i!=j and d < const:
                        A[i][j] = 1
        return A

    def GaussianWeight(self, A):
        W = np.zeros((self.N,self.N))
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(A, v)
            for j, x in enumerate(w):
                if x == 1:
                    W[i][j] = np.exp(-(self.Euclidean(self.data[i], self.data[j])**2)/2)
        return W

    def DegreeMatrix(self,A):
        D = np.zeros((self.N,self.N))
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(A, v)
            D[i][i] = np.sum(w)
        return D

    def Laplacian(self,A,weight='Gaussian'):
        D = self.DegreeMatrix(A)
        if weight == 'Simple':
            W = A
        elif weight == 'Gaussian':
            W = self.GaussianWeight(A)
        return D-W

    def FloydWarshall(self, A):
        G = [[float("inf") for _ in range(self.N)] for _ in range(self.N)]
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(A, v)
            for j, x in enumerate(w):
                if x == 1:
                    G[i][j] = self.Euclidean(self.data[i], self.data[j])
        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    if i == j:
                        G[i][j] = 0
                    G[i][j] =  min(G[i][j], G[i][k]+G[k][j])
        return G


    def MDS(self, G, m):
        I = np.identity(self.N)
        J = np.ones((self.N,self.N))
        H = I-(1/self.N)*J
        D = np.square(G)
        B = -(1/2)*H@D@H

        EigenValues,  EigenFunctions = np.linalg.eig(B)
        L = np.diag(EigenValues[:m])
        L = np.sqrt(L)
        E = EigenFunctions[:,:m]
        X = np.matmul(E,L)

        return X

    def min_max_scale(self, x):
        min_value, max_value = np.min(x,0), np.max(x,0)
        x = (x - min_value) / (max_value - min_value)
        return x

    def plot_data(self, data, color, position, projection):
        data = self.min_max_scale(data)
        if projection == '3d':
            ax = plt.subplot(position, projection = projection)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
            ax.view_init(4, -72)
        elif projection == '2d':
            ax = plt.subplot(position)
            ax.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral)

    def plot_graph(self, data, A, position, projection):
        N = len(data)
        data = self.min_max_scale(data)
        if projection == '3d':
            for i in range(N):
                for j in range(N):
                    v = np.zeros(N)
                    v[i] = 1
                    w = np.zeros(N)
                    w[j] = 1
                    if np.transpose(w)@A@v == 1:
                        ax = plt.subplot(position)
                        ax.plot([data[i][0],data[j][0]], [data[i][1],data[j][1]], [data[i][2],data[j][2]], color='black')
        if projection == '2d':
            for i in range(N):
                for j in range(N):
                    v = np.zeros(N)
                    v[i] = 1
                    w = np.zeros(N)
                    w[j] = 1
                    if np.transpose(w)@A@v == 1:
                        ax = plt.subplot(position)
                        ax.plot([data[i][0],data[j][0]], [data[i][1],data[j][1]], color='black')


