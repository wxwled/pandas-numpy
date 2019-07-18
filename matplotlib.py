# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:27:48 2019

@author: Administrator
"""

#from mpl_toolkits.mplot3d import Axes3D
#import numpy as np
#from matplotlib import pyplot as plt
#fig = plt.figure()
#ax = Axes3D(fig)
#x = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
#y = np.arange(-2 * np.pi, 2 * np.pi, 0.1)
#X, Y = np.meshgrid(x, y)  # 网格的创建，生成二维数组
##X=np.array(x)
##Y=np.array(y)
##print(type(X),X)
##Z = np.sin(X) * np.cos(Y)
##Z=X**4+Y**2
#Z=-X**2-Y**2
##Z=2*X+2*Y
##Z=4*np.sin(X)+Y**2
##Z=np.exp(-1*(X**2+Y**2))
##print(type(Z),Z)
#plt.xlabel('x')
#plt.ylabel('y')
##ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
##plt.contourf(X,Y,Z,cmap="rainbow")
##plt.show()
#


#from matplotlib.tri import (
#    Triangulation, UniformTriRefiner, CubicTriInterpolator)
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import numpy as np
#
#
##-----------------------------------------------------------------------------
## Electrical potential of a dipole
##-----------------------------------------------------------------------------
#def dipole_potential(x, y):
#    """The electric dipole potential V, at position *x*, *y*."""
#    r_sq = x**2 + y**2
#    theta = np.arctan2(y, x)
#    z = np.cos(theta)/r_sq
#    return (np.max(z) - z) / (np.max(z) - np.min(z))
#
#
##-----------------------------------------------------------------------------
## Creating a Triangulation
##-----------------------------------------------------------------------------
## First create the x and y coordinates of the points.
#n_angles = 30
#n_radii = 10
#min_radius = 0.2
#radii = np.linspace(min_radius, 0.95, n_radii)
#
#angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
#angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
#angles[:, 1::2] += np.pi / n_angles
#
#x = (radii*np.cos(angles)).flatten()
#y = (radii*np.sin(angles)).flatten()
#V = dipole_potential(x, y)
#
## Create the Triangulation; no triangles specified so Delaunay triangulation
## created.
#triang = Triangulation(x, y)
#
## Mask off unwanted triangles.
#triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
#                         y[triang.triangles].mean(axis=1))
#                < min_radius)
#
##-----------------------------------------------------------------------------
## Refine data - interpolates the electrical potential V
##-----------------------------------------------------------------------------
#refiner = UniformTriRefiner(triang)
#tri_refi, z_test_refi = refiner.refine_field(V, subdiv=3)
#
##-----------------------------------------------------------------------------
## Computes the electrical field (Ex, Ey) as gradient of electrical potential
##-----------------------------------------------------------------------------
#tci = CubicTriInterpolator(triang, -V)
## Gradient requested here at the mesh nodes but could be anywhere else:
#(Ex, Ey) = tci.gradient(triang.x, triang.y)
#E_norm = np.sqrt(Ex**2 + Ey**2)
#
##-----------------------------------------------------------------------------
## Plot the triangulation, the potential iso-contours and the vector field
##-----------------------------------------------------------------------------
#fig, ax = plt.subplots()
#ax.set_aspect('equal')
## Enforce the margins, and enlarge them to give room for the vectors.
#ax.use_sticky_edges = False
#ax.margins(0.07)
#
#ax.triplot(triang, color='0.8')
#
#levels = np.arange(0., 1., 0.01)
#cmap = cm.get_cmap(name='hot', lut=None)
#ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap=cmap,
#              linewidths=[2.0, 1.0, 1.0, 1.0])
## Plots direction of the electrical vector field
#ax.quiver(triang.x, triang.y, Ex/E_norm, Ey/E_norm,
#          units='xy', scale=10., zorder=3, color='blue',
#          width=0.007, headwidth=3., headlength=4.)
#
#ax.set_title('Gradient plot: an electrical dipole')
#plt.show()


import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.arange(-1 * np.pi, 1 * np.pi, .2), np.arange(-1 * np.pi, 1 * np.pi, .2))
U = np.exp(-1*(X**2+Y**2))*(-2)*X
V = np.exp(-1*(X**2+Y**2))*(-2)*Y
# sphinx_gallery_thumbnail_number = 3

fig3, ax3 = plt.subplots()
ax3.set_title("pivot='tip'; scales with x view")
M = np.hypot(U, V)#斜边长
Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15,cmap='rainbow')
qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
ax3.scatter(X, Y, color='0.5', s=1)

plt.show()