import numpy as np
import matplotlib.pyplot as plt

class ManifoldLearning:
    def __init__(self, data, K, m, adjacency):
        self.data = data
        self.N = len(data)
        self.reg = 0.0001
        self.K = K
        self.m = m
        self.adjacency = adjacency
        self.Adjacency()

    def Euclidean(self, x, y):
        if len(x) == len(y):
            d = 0
            for i in range(len(x)):
                d += ((x[i]-y[i])**2)
            d = np.sqrt(d)
        return d

    def kSmallest(self, l):
        L = [i for i in l]
        Q = [0 for i in range(len(L))]
        while np.count_nonzero(Q) < self.K:
            m = L[0]
            ind = 0
            for i in range(len(L)):
                if L[i] < m:
                    m = L[i]
                    ind = i
            L[ind] = float('inf')
            Q[ind] = m
        return Q

    def Adjacency(self):
        A = np.zeros((self.N,self.N))
        if self.adjacency == 'k-nearest':
            for i in range(self.N):
                L = np.zeros(self.N)
                L[i] = float('inf')
                for j in range(self.N):
                    if i != j:
                        L[j] = self.Euclidean(self.data[i], self.data[j])
                Q = self.kSmallest(L)
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

        elif self.adjacency == 'epsilon-neighborhood':
            for i in range(self.N):
                for j in range(self.N):
                    d = self.Euclidean(self.data[i], self.data[j])
                    if i!=j and d < self.K:
                        A[i][j] = 1
        self.A = A

    def min_max_scale(self, x):
        min_value, max_value = np.min(x,0), np.max(x,0)
        x = (x - min_value) / (max_value - min_value)
        return x

    def plot_data(self, data, color, position, projection, title):
        data = self.min_max_scale(data)
        if projection == '3d':
            ax = plt.subplot(position, projection = projection)
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, cmap=plt.cm.Spectral)
            ax.view_init(4, -72)
        elif projection == '2d':
            ax = plt.subplot(position)
            ax.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral)
        ax.set_title(title)

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


class Iso(ManifoldLearning):
    def __init__(self, data, K, m, adjacency='k-nearest'):
        super().__init__(data, K, m, adjacency)

    def FloydWarshall(self):
        G = [[float("inf") for _ in range(self.N)] for _ in range(self.N)]
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(self.A, v)
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

    def MDS(self):
        I = np.identity(self.N)
        J = np.ones((self.N,self.N))
        H = I-(1/self.N)*J
        D = np.square(self.G)
        B = -(1/2)*H@D@H

        EigenValues,  EigenFunctions = np.linalg.eig(B)
        L = np.diag(EigenValues[:self.m])
        L = np.sqrt(L)
        E = EigenFunctions[:,:self.m]
        X = np.matmul(E,L)

        return X

    def DimensionReduction(self):
        self.G = self.FloydWarshall()
        Y = self.MDS()

        return Y

class LLE(ManifoldLearning):
    def __init__(self, data, K, m, adjacency='k-nearest'):
        super().__init__(data, K, m, adjacency)

    def GramMatrix(self):
        G = np.zeros((self.N,self.K,self.K))
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(self.A, v)
            Z = []

            j = 0
            while len(Z) < self.K:
                if w[j] == 1:
                    Z.append(self.Euclidean(self.data[i],self.data[j]))
                j += 1
            for k in range(self.K):
                for l in range(self.K):
                    G[i][k][l] = Z[k]*Z[l]

            if np.linalg.det(G[i])==0:
                G[i] = G[i] + self.reg*np.identity(self.K)
        return G
            
    def WeightsByLagrangeMult(self):
        w =  np.zeros((len(self.G),self.K))
        for i in range(len(self.G)):
            w[i] = np.linalg.solve(self.G[i], np.ones(self.K))
            w[i] = w[i]/np.sum(w[i])
        return w

    def DimReduct(self):
        W = np.zeros((self.N, self.N))
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(self.A, v)
            k = 0
            for j, x in enumerate(w):
                if x == 1:
                    W[i][j] = self.w[i][k]
                    k += 1
                    if k == len(self.w[0]):
                        break

        I = np.identity(self.N) - W
        M = np.matmul(I.transpose(), I)
        eigenValues, eigenFunctions = np.linalg.eig(M)
        L = len(eigenFunctions[0])
        X = eigenFunctions[:, -2:-2-self.m:-1]

        return X

    def DimensionReduction(self):
        self.G = self.GramMatrix()
        self.w = self.WeightsByLagrangeMult()
        Y = self.DimReduct()

        return Y

class LE(ManifoldLearning):
    def __init__(self, data, K, m, adjacency='k-nearest'):
        super().__init__(data, K, m, adjacency)

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

    def DegreeMatrix(self):
        D = np.zeros((self.N,self.N))
        for i in range(self.N):
            v = np.zeros(self.N)
            v[i] = 1
            w = np.matmul(self.A, v)
            D[i][i] = np.sum(w)
        return D

    def Laplacian(self, weight='Gaussian'):
        self.D = self.DegreeMatrix()
        if weight == 'Simple':
            W = self.A
        elif weight == 'Gaussian':
            W = self.GaussianWeight(self.A)
        self.W = W
        return self.D - self.W

    def DimensionReduction(self):
        L = self.Laplacian()
        eigenValue, eigenFuction = np.linalg.eigh(L)
        Y = eigenFuction[:,:self.m]

        return Y