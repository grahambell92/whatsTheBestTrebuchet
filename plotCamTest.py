import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

verts = [
    (0., 0.), # left, bottom
    (0., 1.), # left, top
    (1., 1.), # right, top
    (1., 0.), # right, bottom
    (0., 0.), # ignored
    ]

import numpy as np
import itertools as it
th0 = 180*np.pi/180
a = 1.0
b = 0.1
thPoints = np.arange(0,2*np.pi, 0.1)
rPoints = [a - b*(th + th0) for th in thPoints]
xPoints = rPoints*np.sin(thPoints)
yPoints = rPoints*np.cos(thPoints)

verts = zip(xPoints,yPoints)
#plt.plot(verts)
#plt.show()
#Define codes as empty numpy array
codes = np.zeros(len(xPoints))
#Put first element as Path.MOVETO
codes[0] = Path.MOVETO
#Put N elements inside codes as Path.LINETO
codes[1:-2] = Path.LINETO
# Put last element as Path.CLOSEPOLY
codes[-1] = Path.CLOSEPOLY
'''
codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
'''
path = Path(verts, codes)

fig = plt.figure()
ax = fig.add_subplot(111)
patch = patches.PathPatch(path, facecolor='orange', lw=2)
ax.add_patch(patch)
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
plt.show()