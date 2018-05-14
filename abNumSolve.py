__author__ = 'Graham'

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
# 19/12/2015 If you plot this equation in desmos with b = x then choose the y-axis intercept you get the correct value for a.
# For some reason this code is not solving the equations correctly.
def equation(b):
    a = 0.4
    y4max = 1.0
    x4 = 1.0

    # New including the theta0 offset
    net = np.sqrt(x4**2 - a**2) - np.sqrt(y4max**2 + x4**2) + (a*np.sqrt(a**2+b**2))/(2*b) + 0.5 * b * np.log10((np.sqrt(a**2+b**2)+a)/(b)) - 0.5*b*np.log10(b)

    #print net
    return net

print(fsolve(equation,0.1))

#net = np.sqrt(x4**2 - a**2) - np.sqrt(y4max**2 + x4**2) + (a*np.sqrt(a**2+b**2))/(2*b) + 0.5 * b * np.log((np.sqrt(a**2+b**2)+a)/(b)) - 0.5*b*np.log(b)

#plt.show()