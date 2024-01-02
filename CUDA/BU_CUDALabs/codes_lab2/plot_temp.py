import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #need this for 3D plots 
# with open('temperature_cpu.txt', 'r') as fp:
#     data = [list(line.strip().split(' ')) for line in fp]

# with open('temperature_texture.txt', 'r') as fp:
#     data = [list(line.strip().split(' ')) for line in fp]

with open('temperature_global.txt', 'r') as fp:
    data = [list(line.strip().split(' ')) for line in fp]   

x = [float(elem[0]) for elem in data]
y = [float(elem[1]) for elem in data]
res = [float(elem[2]) for elem in data]

X = np.array(x)
Y = np.array(y)
RES = np.array(res)


plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),\
                           np.linspace(np.min(Y),np.max(Y),10))
plotz = interp.griddata((X,Y),RES,(plotx,ploty),method='linear')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') #one way to create a 3D plot - this creates axes of type 3D

#fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5) - takes up two rows, three columns, and is at index 5 
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='hot')  
plt.show()