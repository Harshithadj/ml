#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 100
X = np.linspace(-3, 3, num=n)
Y = np.sin(X)
X += np.random.normal(scale=.1, size=n)
plt.scatter(X, Y)

def local_regression(x0, X, Y, tau):
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]
    xw = X.T * radial_kernel(x0, X, tau)
    beta = np.linalg.pinv(xw @ X) @ xw @ Y
    return (x0 @ beta)

def radial_kernel(x0, X, tau):
    return (np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau * tau)))

def plot_lwr(tau):
    domain = np.linspace(-3, 3, num=300)
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.scatter(X, Y, alpha=.3)
    plt.plot(domain, prediction, color='red')
    return plt

plot_lwr(0.04)


# In[6]:


plot_lwr(0.01)


# In[ ]:




