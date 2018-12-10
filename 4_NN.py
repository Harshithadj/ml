#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def reweightingoflevel2():

    derivative_of_error = output[0]-estimated_output[0]

    w2[0][0] = w2[0][0] + 0.002 * (derivative_of_error * hidden_out[0])
    w2[1][0] = w2[1][0] + 0.002 * (derivative_of_error * hidden_out[1])

def reweightingoflevel1():

    w1[0][0] = w1[0][0] + 0.002*(x[0] * hidden_out[0] * w2[0][0])
    w1[0][1] = w1[0][1] + 0.002*(x[0] * hidden_out[1] * w2[1][0])

    w1[1][0] = w1[1][0] + 0.002*(x[1] * hidden_out[0] * w2[0][0])
    w1[1][1] = w1[1][1] + 0.002*(x[1] * hidden_out[1] * w2[1][0])


def relu(a):
    i = 0
    b = [None] * len(a)
    for x in a:
       a[i] = np.maximum(0, x)
       i = i+1
    print(w1)
    return a

# initialization of weights and inputs
w1 = [[0.7, 0.5],
      [0.9, 0.7]]

w2 = [[0.5],
      [0.6]]

x = [2, 5]
output = [1]

for i in range(1000):

    hidden_in = np.matmul(w1, x)
    hidden_out = relu(hidden_in)
    print("hidden out:")
    print(hidden_out.tostring())
    estimated_output = np.matmul(hidden_out, w2)
    print("estimated output")
    print(estimated_output)
    reweightingoflevel2()
    reweightingoflevel1()

    print(w1)


# In[ ]:





# In[ ]:




