#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np

# we want to train the network to give a NOT function, that is if you input 1 it returns 0, and if you input 0 it returns 1.
# First we set the state of the network
σ = np.tanh
w1 = -5
b1 = 5

# Then we define the neuron activation.
def a1(a0) :
  return σ(w1 * a0 + b1)
  
# Finally let's try the network out!
# Replace x with 0 or 1 below,
a1(1)


# In[ ]:




