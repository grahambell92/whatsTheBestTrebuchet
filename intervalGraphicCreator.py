# Short script to utilize the murlin trebuchet class to create the time interval graphic.

# Written by Graham Bell 6th December 2016.

import numpy as np
from optimizeAllTrebs import simpleMurlinDict
import matplotlib.pyplot as plt
import latexPlotTools as lpt

# Perform the throw at m1 = 50
trebDict = simpleMurlinDict
# Create instance of the trebuchet.
treb = trebDict['TrebClass']()

zinit = trebDict['zinit']

m1 = 50.0
a = 1.22
L2 = 4.0
L3 = 4.0
range, throwAngle = treb.runTreb(L2=L2, L3=L3, a=a, m1=m1, endtime=2.0, zinit=zinit, theta0=np.pi/4)
# Throw step index
maxThrowIndex = treb.maxThrowIndex
print('range', range, 'throw Angle', throwAngle, 'maxThrowIndex', maxThrowIndex)
fig = plt.figure(figsize=(10, 6), dpi=80)
plt.subplot(1, 1, 1, aspect='equal')

bounds = 6
xlimits = plt.xlim(-bounds, bounds)
ylimits = plt.ylim(-bounds, bounds)
ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')

fig, ax = plt.subplots()
frameInterval = 5
for frame in np.arange(0, maxThrowIndex, step=frameInterval):
    ax = treb.plotTreb(ax, i=int(frame), projRadius=0.2, CWRadius=0.2, lineThickness=1.0, beamCentreRadius=0.1,
                       imageAlpha=0.2)
# Plot the throw point
ax = treb.plotTreb(ax, i=maxThrowIndex, projRadius=0.2, CWRadius=0.2, lineThickness=1.0, beamCentreRadius=0.1)
bounds = 10
ax.set_xlim(-bounds, bounds)
ax.set_ylim(-bounds, bounds)
ax.set_aspect('equal')
plotName = 'murlinTimeInterval'
lpt.savefig(plotName)
#fig.savefig('./murlinTimeInterval.png', dpi=500)
