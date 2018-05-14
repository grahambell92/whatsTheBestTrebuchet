# Written by Graham Bell 19/07/2016
# Compares the saved optimisations of the different trebuchets to find the greatest
# massRatio/range curve

import pickle
import numpy as np
import latexPlotTools as lpt
import matplotlib.pyplot as plt
import itertools
from trebDicts import FATDict, murlinDict, standardDict, standardWheeledDict, pendulumDict, simpleMurlinDict
import pandas as pd


markers = itertools.cycle(('^', '+', '.', 'o', '*'))
fig, ax = lpt.newfig(0.8)

#scale = 0.5 # correct figure width of the page that you want
#fig_width_pt = 503.169  # Get this from LaTeX using \the\textwidth
#inches_per_pt = 1.0 / 72.27  # Convert pt to inch
#golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
#fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
#fig_height = fig_width * golden_mean # height in inches # added here to increase size to put in box
#fig_size = [fig_width, fig_height + 0.8]

#fig = plt.figure(figsize=fig_size)
#ax = plt.subplot(111)

# murlinPath
trebDicts = [pendulumDict, FATDict, standardDict, standardWheeledDict, simpleMurlinDict]

for trebDict in trebDicts:
    pickleFileName = trebDict['pickleFileName']
    plotLabel = trebDict['trebType']
    print('Reading Name', pickleFileName)
    data = pd.read_pickle(pickleFileName)
    massRatioSpace = data.index.values
    plt.semilogx(massRatioSpace, data['Range']/massRatioSpace, marker=next(markers), markersize=2,
                 linewidth=0.5, markerfacecolor='None', label=plotLabel)

xlimits = plt.xlim(10,1500)
plt.xlabel(r'CW/projectile ($m_{1}/m_{2}$) mass ratio')
plt.ylabel(r'$\frac{\mathrm{Range}}{m_{1}/m_{2}}$')
#box = ax.get_position()
#ax.set_position([box.x0 + box.width*0.1, box.y0 + box.height *0.4, box.width * 0.92, box.height * 0.6])
#ax.spines['top'].set_visible(False)
lgnd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

#plt.tight_layout()
lpt.savefig('rangeRatioComparison', bbox_extra_artists=(lgnd,), bbox_inches='tight')


