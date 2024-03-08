import numpy as np
import matplotlib.pyplot as plt


def inc(x, a, b):  # increasing membership function
    if x <= a:
        r = 0
    elif x >= b:
        r = 1
    else:
        r = (x - a)/(b - a)
    return r


def dec(x, a, b):  # decreasing membership function (the same as increasing, but flipped)
    r = 1 - inc(x, a, b)
    return r


def gauss(x, mu, sig):  # gaussian membership function
    r = np.exp(-(x - mu)**2/(2 * sig**2))
    return r


def sigmoid(x, a, b):  # sigmoid membership function
    r = 1/(1 + np.exp(-a * (x - b)))
    return r


def trapmf(x, params):
    a, b, c, d = params
    if x <= a or x >= d:
        r = 0
    elif a < x <= b:
        r = (x - a) / (b - a)
    elif b < x <= c:
        r = 1
    elif c < x <= d:
        r = (d - x) / (d - c)
    else:
        r = 0
    return r


x = np.linspace(0, 10, 100)
y = np.zeros_like(x)  # create zero array with same dimension as x

for i in range(len(x)):
    y[i] = sigmoid(x[i], -4, 5)
