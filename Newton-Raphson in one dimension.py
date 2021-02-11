#!/usr/bin/env python
# coding: utf-8

# In[1]:


x = 1
def f (x) :
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30

print(f(x))


# In[2]:


def d_f (x) :
  return x**5 - 12*x**3 - 2*x**2 + 27*x + 18

print(d_f (x) )


# In[10]:


import pandas as pd

x = -4.0 # x=1.99 iterate over 15 times, x=3.1 oscillate without settling

d = {"x" : [x], "f(x)": [f(x)]}
for i in range(0, 20):
  x = x - f(x) / d_f(x)
  d["x"].append(x)
  d["f(x)"].append(f(x))

pd.DataFrame(d, columns=['x', 'f(x)'])


# In[11]:


from scipy import optimize

def f (x) :
  return x**6/6 - 3*x**4 - 2*x**3/3 + 27*x**2/2 + 18*x - 30
  
x0 = -5
optimize.newton(f, x0)


# In[ ]:




