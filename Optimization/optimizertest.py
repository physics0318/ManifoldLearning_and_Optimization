import numpy as np
import matplotlib.pyplot as plt

class optimizers:
    def __init__(self, ftn, pos):
        self.epsilon = 0.000001
        self.ftn = ftn
        self.pos = pos
        self.time = 0

        self.historyX = []
        self.historyY = []

    def diff(self, f, i):
        p = self.pos[i]
        q = self.pos[(i-1)**2]
        if i == 0:
            return(f(p+self.epsilon, q)-f(p, q))/self.epsilon
        if i == 1:
            return(f(q, p+self.epsilon)-f(q, p))/self.epsilon

    def updatehist(self):
        self.historyX.append(self.pos[0])
        self.historyY.append(self.pos[1])


class SGD(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.3

    def nextpos(self):
        self.time += 1
        for i in range(2):
           self.pos[i] -= self.lr*self.diff(self.ftn, i)
        self.updatehist()

class Momentum(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.05
        self.v = [0.4, -0.4]

    def nextpos(self):
        self.time += 1
        for i in range(2):
            self.v[i] -= self.lr*self.diff(self.ftn, i)
            self.pos[i] += self.v[i]
        self.updatehist()

class Adagrad(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.3
        self.h = [0,0]

    def nextpos(self):
        self.time += 1
        for i in range(2):
            self.h[i] += self.diff(self.ftn, i)**2
            self.pos[i] -= (self.lr/np.sqrt(self.h[i]))*self.diff(self.ftn, i)
        self.updatehist()

class RMSprop(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.3
        self.h = [0, 0]
        self.rho = 0.4

    def nextpos(self):
        self.time += 1
        for i in range(2):
            self.h[i] = self.rho*self.h[i] + (1-self.rho)*(self.diff(self.ftn, i)**2)
            self.pos[i] -= (self.lr/np.sqrt(self.h[i]))*self.diff(self.ftn, i)
        self.updatehist()

class Adam(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.1
        self.b1 = 0.3
        self.b2 = 0.3
        self.m = [0, 0]
        self.v = [0, 0]

    def nextpos(self):
        self.time += 1
        for i in range(2):
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*self.diff(self.ftn, i)
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(self.diff(self.ftn, i)**2)

            M = self.m[i]/(1-self.b1)
            V = self.v[i]/(1-self.b2)

            self.pos[i] -= self.lr*(M/np.sqrt(V))
        self.updatehist()

class NesterovMomentum(optimizers):
    def __init__(self, ftn, pos):
        super().__init__(ftn, pos)
        self.updatehist()
        self.lr = 0.05
        self.v = [0.4, -0.4]

    def nextpos(self):
        self.time += 1
        
        for i in range(2):
            self.pos[i] += self.v[i]
            self.v[i] -= self.lr*self.diff(self.ftn, i)
        self.updatehist()

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x, y)
f = lambda x, y: (1/20)*(x**2) + y**2
Z = f(X,Y)

posX = 5
posY = 5
pos = [posX, posY]
d = 10
tol = 0.0001
maxTime = 50

optimizer = Momentum(f, pos)

while d > tol and optimizer.time < maxTime:
    optimizer.nextpos()
    t = optimizer.time
    d = (optimizer.historyX[-1]-optimizer.historyX[-2])**2 + (optimizer.historyY[-1]-optimizer.historyY[-2])**2

fig = plt.figure()


ax = plt.axes(projection='3d')
ax.plot(optimizer.historyX, optimizer.historyY, f(np.array(optimizer.historyX), np.array(optimizer.historyY)), color = 'r')
ax.scatter(optimizer.historyX, optimizer.historyY, f(np.array(optimizer.historyX), np.array(optimizer.historyY)), marker = 'o')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()