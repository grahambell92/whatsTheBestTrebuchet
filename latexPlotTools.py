# Script containing the plotting tools for making nice default plots.
# Written by Graham Bell 06/04/2016
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use('pgf')
def customFloatFormat(x):
    return '{:.3f}'.format(x)

def saveToCSV(pickleFilename, dataFrame):
    # Write out data as csv for latex tables
    newDataFrame = dataFrame.applymap(customFloatFormat)
    newDataFrame.to_csv(pickleFilename, float_format='%.3f')
    return

def figsize(scale):
    fig_width_pt = 503.169                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "axes.linewidth": 0.7,              # axes border width
    "patch.linewidth" : 0.0,            # legend border width
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "legend.numpoints" : 3,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.size": 3,
    "xtick.minor.size": 2,
    "ytick.major.size": 3,
    "ytick.minor.size": 2,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, bbox_extra_artists=None, bbox_inches=None):
    plt.savefig('{}.pgf'.format(filename), bbox_extra_artists=bbox_extra_artists, bbox_inches=bbox_inches)
    plt.savefig('{}.pdf'.format(filename), bbox_extra_artists=bbox_extra_artists, bbox_inches=bbox_inches)

#tableau10 in rgb
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# reformatted for matplotlib
factor = 0.85
tableau10 = [(r*factor / 255., g*factor / 255., b*factor / 255.) for (r, g, b) in tableau20[::2]]
tableau20 = [(r / 255., g / 255., b / 255.) for (r, g, b) in tableau20]

markers = ['o', 's', '8', 'v', '^', '<', '>',  'p', '*', 'h', 'H', 'D', 'd']

if __name__ == "__main__":
    fig, ax = newfig(0.5)
    for index, colour in enumerate(tableau10):
        plt.plot(index, marker = '.', color = colour)
    savefig("Testplot")