#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First we define the functions,
def f (x, y) :
    return np.exp(-(2*x*x + y*y - x*y) / 2)

def g (x, y) :
    return x*x + 3*(y+1)**2 - 1

# Next their derivatives,
def dfdx (x, y) :
    return 1/2 * (-4*x + y) * f(x, y)

def dfdy (x, y) :
    return 1/2 * (x - 2*y) * f(x, y)

def dgdx (x, y) :
    return 2*x

def dgdy (x, y) :
    return 6 * (y+1)


# In[3]:


import numpy as np
from scipy import optimize

def DL (xyλ) :
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            - g(x, y)
        ])

(x0, y0, λ0) = (-1, -1, 0)
x, y, λ = optimize.root(DL, [x0, y0, λ0]).x　
    #optimize.root  Find a root of a vector function.
    #(x0, y0, λ0) を起点にDL関数の極小値を求める
print("x = %g" % x) 
    #%gは浮動小数点　https://www.javadrive.jp/python/string/index23.html#section1
    #右の"% x"は、左の％に代入する変数　https://programming-study.com/technology/python-print/
print("y = %g" % y)
print("λ = %g" % λ)
print("f(x, y) = %g" % f(x, y))


# In[5]:





# In[7]:


x = -0.958963
y = -1.1637
λ = -0.246538

g (x, y)


# In[ ]:




