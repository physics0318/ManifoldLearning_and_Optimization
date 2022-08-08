import numpy as np

def GD(layer, error, output, input):
    lr = 0.1
    layer =layer + lr*np.dot((error*output*(1.0 - output)), np.transpose(input))
    return layer

def Momentum(layer, error, output, input, counter, v):
    a = 0.1
    lr = 0.1
    if counter == 1:
        v = layer
    v = a*v + lr*np.dot((error*output*(1.0 - output)), np.transpose(input))
    layer = layer + v
    return v, layer

def Adagrad(layer, error, output, input, counter, h):
    lr = 0.1
    if counter == 1:
        h = layer
    h = h + np.square(np.dot((error*output*(1.0 - output)), np.transpose(input)))
    layer = layer - lr*(np.sqrt(np.reciprocal(h)), np.dot((error*output*(1.0 - output)), np.transpose(input)))
    return h, layer

def Adam(layer, error, output, input, counter, m, v):
    a = 0.1
    b = 0.1
    lr = 0.1
    if counter == 1:
        m = layer
        v = layer
    m = a*m + (1-a)*np.dot((error*output*(1.0 - output)), np.transpose(input))
    v = b*v + (1-b)*np.square(np.dot((error*output*(1.0 - output)), np.transpose(input)))
    M = m/(1-a)
    V = v/(1-b)
    layer = layer - lr*(M/np.sqrt(V))
    return m, v, layer
