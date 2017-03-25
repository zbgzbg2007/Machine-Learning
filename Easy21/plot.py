from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

'''
This file plot the results from different methods.

'''

'''
# plot the optimal value function
name = 'optimal-Q' 
data = []
q = dict()
with open(name, 'r') as file1:
  for l in file1:
    nums = l.split()
    q[((int(nums[0]), int(nums[1])), nums[2])] = float(nums[3])
      
for s, a in q:
  i, j = s
  data.append((i, j, max(q[(s, 'h')], q[(s, 's')])))

x, y, z = zip(*data)
z = map(float, z)
grid_x, grid_y = np.mgrid[min(x):max(x):400j, min(y):max(y):400j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
ax.set_xlabel('dealer')
ax.set_ylabel('player')
ax.set_zlabel('value function')

plt.show()
'''

'''
# plot the mean-squared error against lambda

#name = 'lambdas-errors'
name = 'lambdas-errors-FA'
X = [0.1 * i for i in range(11)]
Y = list()
with open(name, 'r') as file2:
  for l in file2:
    nums = l.split()
    Y.append(float(nums[1]))

X = np.linspace(0, 1, 11, endpoint=True)


fig, ax = plt.subplots()
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'mean-squared error')
ax.plot(X, Y, '--')
ax.plot(X, Y, 'ro')
ax.set_xlim((-0.1, 1.1))
#ax.set_ylim((20, 40))
ax.set_ylim((15, 35))

plt.show()

'''
# plot the learning curve of the mean-squared error against episode number.
#name = 'learning-curve'
name = 'learning-curve-FA'
#X = [5000 * (i+1) for i in range(20)]
X = [1000 * (i+1) for i in range(20)]
Y1, Y2 = list(), list()
with open(name, 'r') as file3:
  for l in file3:
    nums = l.split()
    Y1.append(float(nums[1]))
    Y2.append(float(nums[2]))

#X = np.linspace(5000, 105000, 20, endpoint=True)
X = np.linspace(1000, 21000, 20, endpoint=True)


fig, ax = plt.subplots()
ax.set_xlabel('number of iterations')
ax.set_ylabel('mean-squared error')
ax.plot(X, Y1, '-', label='$\lambda = 0$', linewidth=2)
ax.plot(X, Y2, 'r-', label='$\lambda = 1$', linewidth=2)
ax.legend()

plt.show()
