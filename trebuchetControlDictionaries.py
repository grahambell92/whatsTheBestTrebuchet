
# This python file contains the different dictionaries to control the parametric and optimisation throws.
# It allow the better abstraction of the parametric and optimisation scripts.
# The optimisation and parametric scripts reference this file

import numpy as np
from FATTrebClass import FATTreb
from standardTrebOnWheelClass import standardTrebOnWheels
from standardTrebClass import standardTreb


#---- Trebuchet Dictionaries -----------
FATDict = {'pickleFileName': 'ParametricFATTreb.dat',
           'plotName': 'ParametricFATContour',
           'TrebClass': FATTreb('TestTreb'),
           'L2sMin': 0.5,
           'L2sMax': 8,
           'L3sMin': 0.5,
           'L3sMax': 8,
           'm1': 100,
           'm3': 1,
           'zinit' : np.array([45.0 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0])
           }