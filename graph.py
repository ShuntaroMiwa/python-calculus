#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

theta = np.linspace(-np.pi, np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
z = 2*x+3*y

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(x,y,z)

ax.scatter(2/np.sqrt(13),3/np.sqrt(13),np.sqrt(13))
ax.scatter(-2/np.sqrt(13),-3/np.sqrt(13),-np.sqrt(13))

plt.figure(figsize=(10, 8)) #pythonスクリプトとjupyterとでは、matplotlibの図のサイズが違うので注意

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z = -1 * np.exp(x - y*y + x*y)

plt.figure(figsize=(10, 8)) #pythonスクリプトとjupyterとでは、matplotlibの図のサイズが違うので注意https://linus-mk.hatenablog.com/entry/2019/09/08/232147

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x,y,z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z = np.cosh(y) + x -2

plt.figure(figsize=(10, 8)) #pythonスクリプトとjupyterとでは、matplotlibの図のサイズが違うので注意https://linus-mk.hatenablog.com/entry/2019/09/08/232147

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(x,y,z)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:




