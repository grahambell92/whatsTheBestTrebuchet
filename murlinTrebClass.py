# Class for the Murlin Trebuchet.
# Created by Graham Bell 16/06/2016

import numpy as np
from numpy import sin, cos, pi, log, sqrt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import animation


def solveAandB(b, a , x4):
    # Enter in an a and x4 value and b is numerically solved.
    # Max Drop
    y4max = 1.0 + np.sin(np.pi/4)
    C1 = (a * np.sqrt(a ** 2 + b ** 2)) / (2 * b) + 0.5 * b * np.log((np.sqrt(a ** 2 + b ** 2) + a) / (b))
    net = np.sqrt(x4 ** 2 - a ** 2) - np.sqrt(y4max ** 2 + x4 ** 2) - 0.5 * b * np.log(b) + C1
    return net

class murlinTreb:
    def __init__(self,name='treb1', beamDensity = 1.0, g=9.81):
        self.name = name
        self.beamDensity = beamDensity
        self.g = g

    def L4Length(self, theta, a, b, x4):
        L4 = np.sqrt(x4 ** 2 - a ** 2) + (a * np.sqrt(a ** 2 + b ** 2)) / (2 * b) + b * np.log(np.sqrt(a ** 2 + b ** 2) + a) / 2 - ((a - b * (theta)) * np.sqrt((a - b * (theta)) ** 2 + b ** 2)) / (2 * b) - b * np.log(np.sqrt((a - b * (theta)) ** 2 + b ** 2) + (a - b * (theta))) / 2
        return L4

    def dL4Length(self, theta, a, b, x4):
        dL4 = -b * (-b * (a - b * theta) / np.sqrt(b ** 2 + (a - b * theta) ** 2) - b) / (2 * (a - b * theta + np.sqrt(b ** 2 + (a - b * theta) ** 2))) + (a - b * theta) ** 2 / (2 * np.sqrt(b ** 2 + (a - b * theta) ** 2)) + np.sqrt(b ** 2 + (a - b * theta) ** 2) / 2
        return dL4

    def ddL4Length(self, theta, a, b, x4):
        ddL4 = b * (b ** 2 * ((a - b * theta) / np.sqrt(b ** 2 + (a - b * theta) ** 2) + 1) ** 2 / (a - b * theta + np.sqrt(b ** 2 + (a - b * theta) ** 2)) ** 2 + b ** 2 * ((a - b * theta) ** 2 / (b ** 2 + (a - b * theta) ** 2) - 1) / (np.sqrt(b ** 2 + (a - b * theta) ** 2) * (a - b * theta + np.sqrt(b ** 2 + (a - b * theta) ** 2))) + (a - b * theta) ** 3 / (b ** 2 + (a - b * theta) ** 2) ** (3 / 2) - 3 * (a - b * theta) / np.sqrt(b ** 2 + (a - b * theta) ** 2)) / 2
        return ddL4

    def deriv(self, z, t):
        # z[0] = theta
        # z[1] = alpha
        # z[2] = theta_dot
        # z[3] = alpha_dot
        g = self.g
        L2 = self.L2
        L3 = self.L3
        x4 = self.x4
        m1 = self.m1
        m3 = self.m3
        m_b = self.m_b
        L4 = self.L4Length(z[0], self.a, self.b, self.x4)
        dL4 = self.dL4Length(z[0], self.a, self.b, self.x4)
        ddL4 = self.ddL4Length(z[0], self.a, self.b, self.x4)
        L1 = self.a - self.b * z[0]
        dL1 = -self.b
        ddL1 = 0
        dz0 = z[2]
        dz1 = z[3]
        dz2 = 3 * (-L2 ** 2 * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 + 2 * L2 ** 2 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 + 2 * L2 ** 2 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 - L2 ** 2 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 - 2 * L2 ** 2 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 - L2 ** 2 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(2 * z[1] - 2 * z[0]) * z[2] ** 2 - 2 * L2 * L3 * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1] - z[0]) * z[3] ** 2 + 4 * L2 * L3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - z[0]) * z[3] ** 2 + 4 * L2 * L3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - z[0]) * z[3] ** 2 - 2 * L2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - z[0]) * z[3] ** 2 - 4 * L2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - z[0]) * z[3] ** 2 - 2 * L2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - z[0]) * z[3] ** 2 - L2 * g * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.sin(2 * z[1] - z[0]) - L2 * g * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.sin(z[0]) + 2 * L2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(2 * z[1] - z[0]) + 2 * L2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(z[0]) + 2 * L2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(2 * z[1] - z[0]) + 2 * L2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(z[0]) - L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(2 * z[1] - z[0]) - L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(z[0]) - 2 * L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(2 * z[1] - z[0]) - 2 * L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(z[0]) - L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(2 * z[1] - z[0]) - L2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(z[0]) - L2 * g * m_b * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.sin(z[0]) + 2 * L2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(z[0]) + 2 * L2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(z[0]) - L2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(z[0]) - 2 * L2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(z[0]) - L2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(z[0]) + 2 * g * m1 * x4 ** 4 * L1 * dL1 + 2 * g * m1 * x4 ** 4 * L4 * dL4 - 4 * g * m1 * x4 ** 2 * L1 ** 3 * dL1 - 4 * g * m1 * x4 ** 2 * L1 ** 2 * L4 * dL4 - 4 * g * m1 * x4 ** 2 * L1 * L4 ** 2 * dL1 - 4 * g * m1 * x4 ** 2 * L4 ** 3 * dL4 + 2 * g * m1 * L1 ** 5 * dL1 + 2 * g * m1 * L1 ** 4 * L4 * dL4 + 4 * g * m1 * L1 ** 3 * L4 ** 2 * dL1 + 4 * g * m1 * L1 ** 2 * L4 ** 3 * dL4 + 2 * g * m1 * L1 * L4 ** 4 * dL1 + 2 * g * m1 * L4 ** 5 * dL4 + 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * z[2] ** 2 * ddL1 * dL1 + 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * z[2] ** 2 * ddL1 * dL4 + 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * z[2] ** 2 * ddL4 * dL1 - 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * dL1 * z[2] ** 2 * dL1 ** 2 - 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * dL4 * z[2] ** 2 * dL1 * dL4 + 4 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * z[2] ** 2 * dL1 ** 3 + 4 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * z[2] ** 2 * dL1 * dL4 ** 2 + 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * z[2] ** 2 * ddL4 * dL4 - 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * dL1 * z[2] ** 2 * dL1 * dL4 - 2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * dL4 * z[2] ** 2 * dL4 ** 2 + 4 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * z[2] ** 2 * dL1 ** 2 * dL4 + 4 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * z[2] ** 2 * dL4 ** 3 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * z[2] ** 2 * ddL1 * dL1 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * z[2] ** 2 * ddL1 * dL4 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * z[2] ** 2 * ddL4 * dL1 + 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * dL4 * z[2] ** 2 * dL1 * dL4 - 4 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * z[2] ** 2 * dL1 * dL4 ** 2 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * z[2] ** 2 * ddL1 * dL1 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * z[2] ** 2 * ddL4 * dL4 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * dL1 * z[2] ** 2 * dL1 * dL4 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * dL4 * z[2] ** 2 * dL1 ** 2 + 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * dL4 * z[2] ** 2 * dL4 ** 2 + 8 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * z[2] ** 2 * dL1 ** 2 * dL4 - 4 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * z[2] ** 2 * dL4 ** 3 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * z[2] ** 2 * ddL1 * dL4 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * z[2] ** 2 * ddL4 * dL1 + 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * dL1 * z[2] ** 2 * dL1 ** 2 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * dL1 * z[2] ** 2 * dL4 ** 2 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * dL4 * z[2] ** 2 * dL1 * dL4 - 4 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * z[2] ** 2 * dL1 ** 3 + 8 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * z[2] ** 2 * dL1 * dL4 ** 2 - 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * z[2] ** 2 * ddL4 * dL4 + 2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 3 * dL1 * z[2] ** 2 * dL1 * dL4 - 4 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 3 * z[2] ** 2 * dL1 ** 2 * dL4) / (2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * (-3 * L2 ** 2 * m3 * x4 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) + 3 * L2 ** 2 * m3 * x4 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) + 3 * L2 ** 2 * m3 * x4 ** 4 * np.sin(z[0]) ** 2 + 3 * L2 ** 2 * m3 * x4 ** 4 * np.cos(z[0]) ** 2 + 6 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) - 6 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) - 6 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 * np.sin(z[0]) ** 2 - 6 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 * np.cos(z[0]) ** 2 + 6 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) - 6 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) - 6 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 * np.sin(z[0]) ** 2 - 6 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 * np.cos(z[0]) ** 2 - 3 * L2 ** 2 * m3 * L1 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) + 3 * L2 ** 2 * m3 * L1 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) + 3 * L2 ** 2 * m3 * L1 ** 4 * np.sin(z[0]) ** 2 + 3 * L2 ** 2 * m3 * L1 ** 4 * np.cos(z[0]) ** 2 - 6 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) + 6 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) + 6 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 * np.sin(z[0]) ** 2 + 6 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 * np.cos(z[0]) ** 2 - 3 * L2 ** 2 * m3 * L4 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[1]) * np.cos(z[0]) + 3 * L2 ** 2 * m3 * L4 ** 4 * np.sin(z[1] - z[0]) * np.sin(z[0]) * np.cos(z[1]) + 3 * L2 ** 2 * m3 * L4 ** 4 * np.sin(z[0]) ** 2 + 3 * L2 ** 2 * m3 * L4 ** 4 * np.cos(z[0]) ** 2 + L2 ** 2 * m_b * x4 ** 4 - 2 * L2 ** 2 * m_b * x4 ** 2 * L1 ** 2 - 2 * L2 ** 2 * m_b * x4 ** 2 * L4 ** 2 + L2 ** 2 * m_b * L1 ** 4 + 2 * L2 ** 2 * m_b * L1 ** 2 * L4 ** 2 + L2 ** 2 * m_b * L4 ** 4 - 3 * m1 * x4 ** 2 * L1 ** 2 * dL1 ** 2 - 6 * m1 * x4 ** 2 * L1 * L4 * dL1 * dL4 - 3 * m1 * x4 ** 2 * L4 ** 2 * dL4 ** 2 + 3 * m1 * L1 ** 4 * dL1 ** 2 + 6 * m1 * L1 ** 3 * L4 * dL1 * dL4 + 3 * m1 * L1 ** 2 * L4 ** 2 * dL1 ** 2 + 3 * m1 * L1 ** 2 * L4 ** 2 * dL4 ** 2 + 6 * m1 * L1 * L4 ** 3 * dL1 * dL4 + 3 * m1 * L4 ** 4 * dL4 ** 2))

        dz3 = (6 * L2 ** 3 * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1] - z[0]) * z[2] ** 2 - 12 * L2 ** 3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 - 12 * L2 ** 3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 + 6 * L2 ** 3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 + 12 * L2 ** 3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 + 6 * L2 ** 3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 + 2 * L2 ** 3 * m_b * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1] - z[0]) * z[2] ** 2 - 4 * L2 ** 3 * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 - 4 * L2 ** 3 * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 + 2 * L2 ** 3 * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 + 4 * L2 ** 3 * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 + 2 * L2 ** 3 * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 + 3 * L2 ** 2 * L3 * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 - 6 * L2 ** 2 * L3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 - 6 * L2 ** 2 * L3 * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 + 3 * L2 ** 2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 + 6 * L2 ** 2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 + 3 * L2 ** 2 * L3 * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(2 * z[1] - 2 * z[0]) * z[3] ** 2 + 3 * L2 ** 2 * g * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1] - 2 * z[0]) + 3 * L2 ** 2 * g * m3 * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1]) - 6 * L2 ** 2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - 2 * z[0]) - 6 * L2 ** 2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1]) - 6 * L2 ** 2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - 2 * z[0]) - 6 * L2 ** 2 * g * m3 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1]) + 3 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - 2 * z[0]) + 3 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1]) + 6 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - 2 * z[0]) + 6 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1]) + 3 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - 2 * z[0]) + 3 * L2 ** 2 * g * m3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1]) + 3 * L2 ** 2 * g * m_b * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1] - 2 * z[0]) / 2 + L2 ** 2 * g * m_b * x4 ** 4 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * np.cos(z[1]) / 2 - 3 * L2 ** 2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - 2 * z[0]) - L2 ** 2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1]) - 3 * L2 ** 2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - 2 * z[0]) - L2 ** 2 * g * m_b * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1]) + 3 * L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - 2 * z[0]) / 2 + L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1]) / 2 + 3 * L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - 2 * z[0]) + L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1]) + 3 * L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - 2 * z[0]) / 2 + L2 ** 2 * g * m_b * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1]) / 2 - 6 * L2 * g * m1 * x4 ** 4 * L1 * np.sin(z[1] - z[0]) * dL1 - 6 * L2 * g * m1 * x4 ** 4 * L4 * np.sin(z[1] - z[0]) * dL4 + 12 * L2 * g * m1 * x4 ** 2 * L1 ** 3 * np.sin(z[1] - z[0]) * dL1 + 12 * L2 * g * m1 * x4 ** 2 * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * dL4 + 12 * L2 * g * m1 * x4 ** 2 * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * dL1 + 12 * L2 * g * m1 * x4 ** 2 * L4 ** 3 * np.sin(z[1] - z[0]) * dL4 - 6 * L2 * g * m1 * L1 ** 5 * np.sin(z[1] - z[0]) * dL1 - 6 * L2 * g * m1 * L1 ** 4 * L4 * np.sin(z[1] - z[0]) * dL4 - 12 * L2 * g * m1 * L1 ** 3 * L4 ** 2 * np.sin(z[1] - z[0]) * dL1 - 12 * L2 * g * m1 * L1 ** 2 * L4 ** 3 * np.sin(z[1] - z[0]) * dL4 - 6 * L2 * g * m1 * L1 * L4 ** 4 * np.sin(z[1] - z[0]) * dL1 - 6 * L2 * g * m1 * L4 ** 5 * np.sin(z[1] - z[0]) * dL4 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL1 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL4 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL1 - 12 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 + 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL1 ** 2 + 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL1 * dL4 - 12 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 ** 3 - 12 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 ** 2 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL4 - 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL4 ** 2 + 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL1 * dL4 + 6 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL4 ** 2 - 12 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 * dL4 - 12 * L2 * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL4 ** 3 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL1 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL4 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL1 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 - 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL1 * dL4 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL1 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL4 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL4 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL1 * dL4 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL1 ** 2 - 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL4 ** 2 - 24 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 * dL4 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL4 ** 3 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL1 * dL4 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL1 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 - 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL1 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL4 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * dL4 * z[2] ** 2 * dL1 * dL4 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 ** 3 - 24 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 2 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 * dL4 ** 2 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.sin(z[1] - z[0]) * z[2] ** 2 * ddL4 * dL4 + 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1] - z[0]) * z[2] ** 2 * dL4 ** 2 - 6 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 3 * np.sin(z[1] - z[0]) * dL1 * z[2] ** 2 * dL1 * dL4 + 12 * L2 * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 3 * np.sin(z[1] - z[0]) * z[2] ** 2 * dL1 ** 2 * dL4 - 6 * g * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * np.cos(z[1]) * dL1 ** 2 - 12 * g * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 * np.cos(z[1]) * dL1 * dL4 - 6 * g * m1 * x4 ** 2 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 2 * np.cos(z[1]) * dL4 ** 2 + 6 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 4 * np.cos(z[1]) * dL1 ** 2 + 12 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 3 * L4 * np.cos(z[1]) * dL1 * dL4 + 6 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1]) * dL1 ** 2 + 6 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 ** 2 * L4 ** 2 * np.cos(z[1]) * dL4 ** 2 + 12 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L1 * L4 ** 3 * np.cos(z[1]) * dL1 * dL4 + 6 * g * m1 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * L4 ** 4 * np.cos(z[1]) * dL4 ** 2) / (2 * L3 * np.sqrt(-x4 ** 2 + L1 ** 2 + L4 ** 2) * (3 * L2 ** 2 * m3 * x4 ** 4 * np.cos(2 * z[1] - 2 * z[0]) / 2 + 3 * L2 ** 2 * m3 * x4 ** 4 / 2 - 3 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 * np.cos(2 * z[1] - 2 * z[0]) - 3 * L2 ** 2 * m3 * x4 ** 2 * L1 ** 2 - 3 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 * np.cos(2 * z[1] - 2 * z[0]) - 3 * L2 ** 2 * m3 * x4 ** 2 * L4 ** 2 + 3 * L2 ** 2 * m3 * L1 ** 4 * np.cos(2 * z[1] - 2 * z[0]) / 2 + 3 * L2 ** 2 * m3 * L1 ** 4 / 2 + 3 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 * np.cos(2 * z[1] - 2 * z[0]) + 3 * L2 ** 2 * m3 * L1 ** 2 * L4 ** 2 + 3 * L2 ** 2 * m3 * L4 ** 4 * np.cos(2 * z[1] - 2 * z[0]) / 2 + 3 * L2 ** 2 * m3 * L4 ** 4 / 2 + L2 ** 2 * m_b * x4 ** 4 - 2 * L2 ** 2 * m_b * x4 ** 2 * L1 ** 2 - 2 * L2 ** 2 * m_b * x4 ** 2 * L4 ** 2 + L2 ** 2 * m_b * L1 ** 4 + 2 * L2 ** 2 * m_b * L1 ** 2 * L4 ** 2 + L2 ** 2 * m_b * L4 ** 4 - 3 * m1 * x4 ** 2 * L1 ** 2 * dL1 ** 2 - 6 * m1 * x4 ** 2 * L1 * L4 * dL1 * dL4 - 3 * m1 * x4 ** 2 * L4 ** 2 * dL4 ** 2 + 3 * m1 * L1 ** 4 * dL1 ** 2 + 6 * m1 * L1 ** 3 * L4 * dL1 * dL4 + 3 * m1 * L1 ** 2 * L4 ** 2 * dL1 ** 2 + 3 * m1 * L1 ** 2 * L4 ** 2 * dL4 ** 2 + 6 * m1 * L1 * L4 ** 3 * dL1 * dL4 + 3 * m1 * L4 ** 4 * dL4 ** 2))

        return np.array([dz0, dz1, dz2, dz3])
    def throwAngle(self, alpha):
        return 3 * np.pi / 2 - alpha

    def throwRange(self, theta, thetadot, alpha, alphadot):
        L2 = self.L2
        L3 = self.L3
        h = self.h
        g = self.g
        x2 = -L2 * np.sin(theta)
        y2 = -L2 * np.cos(theta)
        x3 = x2 + L3 * np.cos(alpha)
        y3 = y2 - L3 * np.sin(alpha)

        #gamma = 3 * pi / 2 - alpha
        Vx = L2 * thetadot * cos(pi - theta) + L3 * alphadot * cos(self.throwAngle(alpha))
        Vy = L2 * thetadot * sin(pi - theta) + L3 * alphadot * sin(self.throwAngle(alpha))

        Sx = Vx * ((Vy + np.sqrt((Vy ** 2) + (2 * self.g * (y3 - self.h)))) / self.g) + x3
        if np.isnan(Sx):
            Sx = np.nan
        # theta > pi/2 and theta < 3*pi/2 and
        if Vx > 0 and Vy > 0 and Sx > 0 and alpha < 2 * pi and y3 > 0:  # and thetadot > 0:
            # print "Vx: %.3f" % Vx
            # print "Vy: %.3f" % Vy
            return Sx
        else:
            return 0.01

    def runTreb(self, m1=100, m3=1.0,L2=2.0, L3=2.0, x4=0.8+0.2, a=0.8, b=0.3, endtime=2.0, dt=0.01,
                zinit=np.array([0.01*np.pi/180, 0.1*np.pi/180, 0.0, 0.0])):

        self.m1 = m1
        self.m3 = m3

        self.a = a
        self.b = b

        self.L2 = L2
        self.L3 = L3
        self.h = -np.cos(45 * np.pi / 180) * self.L2
        self.x4 = x4
        self.m_b = self.beamDensity * (self.L3)

        self.endtime = endtime
        self.dt = dt

        self.t = np.linspace(0, self.endtime, self.endtime / self.dt)

        self.zinit = zinit
        self.z = odeint(self.deriv, self.zinit, self.t)

        self.L1 = a - b * (self.z[:, 0])

        self.xb = -self.L2 / 2 * np.sin(self.z[:, 0])
        self.yb = -self.L2 / 2 * np.cos(self.z[:, 0])

        self.x2 = -self.L2 * np.sin(self.z[:, 0])
        self.y2 = -self.L2 * np.cos(self.z[:, 0])

        self.x3 = self.x2 + self.L3 * np.cos(self.z[:, 1])
        self.y3 = self.y2 - self.L3 * np.sin(self.z[:, 1])

        # Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot
        Sx = [self.throwRange(self.z[i, 0], self.z[i, 2], self.z[i, 1], self.z[i, 3]) for i in range(len(self.t))]
        Sx = []
        addNanOnly = False
        for i,t in enumerate(self.t):
            if self.z[i,0] > 4*np.pi/3 or addNanOnly == True:
                Sx.append(np.nan)
                addNanOnly = True
            else:
                Sx.append(self.throwRange(self.z[i, 0], self.z[i, 2], self.z[i, 1], self.z[i, 3]))
        Sx = np.asarray(Sx)
        self.SxMaxThrowAngle = self.throwAngle(self.z[np.nanargmax(Sx), 1])
        rangethrown = np.nanmax(Sx)
        return rangethrown, self.SxMaxThrowAngle

    def init(self):
        return []

    def animate(self, i):
        patches = []

        # Max length of L4 before bouncing up.
        L4turningPoint = self.L4Length(self.a / self.b, self.a, self.b, self.x4)
        # Initialize L4
        #L4 = np.zeros(len(z[:, 0]), dtype=np.float)

        L4 = self.L4Length(self.z[i, 0], self.a, self.b, self.x4)

        if self.z[i, 0] > self.a / self.b:
            L4 = self.L4Length(self.a / self.b, self.a, self.b, self.x4) - (
            self.L4Length(self.z[i, 0], self.a, self.b, self.x4) - self.L4Length(self.a / self.b, self.a, self.b, self.x4))

        else:
            L4 = self.L4Length(self.z[i, 0], self.a, self.b, self.x4)

        y4 = -np.sqrt(L4 ** 2 + self.L1 ** 2 - self.x4 ** 2)
        # Solving the tangent point for where L1 attaches to the cam
        delta = np.arctan2(y4, self.x4)
        beta = np.arccos(self.L1 / self.x4)
        xL1 = self.L1 * np.cos(beta + delta)
        yL1 = self.L1 * np.sin(beta + delta)

        # Vertical and horizontal lines:
        # Horizontal rail
        patches.append(ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=1.7, alpha=0.3, color='k', linestyle='--')))
        # Vertical Rail
        patches.append(ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=1.7, alpha=0.3, color='k', linestyle='--')))

        # Vertical Rail for CW
        patches.append(ax.add_line(plt.Line2D((self.x4, self.x4), (-3, 0), lw=1.7, alpha=0.3, color='k', linestyle='--')))

        # 0, 0 - x2,y2 line
        patches.append(ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=2.5)))

        # x2,y2-x3,y3 line
        patches.append(ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=2.5)))

        # Draw string
        # End 1 is Cam, end 2 is CW
        patches.append(ax.add_line(plt.Line2D((xL1[i % len(self.t)], self.x4), (yL1[i % len(self.t)],
                                                                                y4[i % len(self.t)]), lw=1.8)))

        # m1 CW circle
        patches.append(ax.add_patch(plt.Circle((self.x4, y4[i % len(self.t)]), 0.2, color='k')))

        # Centre Cam Circle
        patches.append(ax.add_patch(plt.Circle((0, 0), self.L1[i % len(self.t)], color='r')))
        # Draw some arrows on that rotate
        LenLine = 0.6
        LineOffset = 0.0
        patches.append(ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + LineOffset), LenLine * np.sin(self.z[i, 0] + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + LineOffset), LenLine * np.cos(self.z[i, 0] + LineOffset)), color='k', linestyle='--')))
        patches.append(ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset)), color='k', linestyle='--')))

        # beam centre circle
        patches.append(ax.add_patch(plt.Circle((self.xb[i % len(self.t)], self.yb[i % len(self.t)]), 0.05, color='k')))

        # Projectile circle
        patches.append(ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.1, color='r')))

        #patches.append(ax.text(-3, 3, "Total Energy: %.3f" % totalEnergy[i]))
        #patches.append(ax.text(-3, 1, "T: %.3f" % T[i]))
        #patches.append(ax.text(-3, 2, "V: %.3f" % V[i]))

        patches.append(ax.text(-3, 2, "Y4: %.3f" % y4[0]))
        #patches.append(ax.text(-3, 2, "Range: %.3f" % Sx[i]))
        # patches.append(ax.add_patch(plt.text(0, 0, "Circle", size=30)))

        return patches

    def plotTreb(self, ax, i):
        lineThickness = 0.4
        # Max length of L4 before bouncing up.
        L4turningPoint = self.L4Length(self.a / self.b, self.a, self.b, self.x4)
        # Initialize L4
        #L4 = np.zeros(len(z[:, 0]), dtype=np.float)

        L4 = self.L4Length(self.z[i, 0], self.a, self.b, self.x4)

        if self.z[i, 0] > self.a / self.b:
            L4 = self.L4Length(self.a / self.b, self.a, self.b, self.x4) - (
            self.L4Length(self.z[i, 0], self.a, self.b, self.x4) - self.L4Length(self.a / self.b, self.a, self.b, self.x4))

        else:
            L4 = self.L4Length(self.z[i, 0], self.a, self.b, self.x4)

        y4 = -np.sqrt(L4 ** 2 + self.L1 ** 2 - self.x4 ** 2)
        # Solving the tangent point for where L1 attaches to the cam
        delta = np.arctan2(y4, self.x4)
        beta = np.arccos(self.L1 / self.x4)
        xL1 = self.L1 * np.cos(beta + delta)
        yL1 = self.L1 * np.sin(beta + delta)

        # Vertical and horizontal lines:
        # Horizontal rail
        ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=lineThickness, alpha=0.3, color='k', linestyle='--'))
        # Vertical Rail
        ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=lineThickness, alpha=0.3, color='k', linestyle='--'))

        # Vertical Rail for CW
        ax.add_line(plt.Line2D((self.x4, self.x4), (-3, 0), lw=lineThickness, alpha=0.3, color='k', linestyle='--'))

        # 0, 0 - x2,y2 line
        ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=lineThickness))

        # x2,y2-x3,y3 line
        ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=lineThickness))

        # Draw string
        # End 1 is Cam, end 2 is CW
        ax.add_line(plt.Line2D((xL1[i % len(self.t)], self.x4), (yL1[i % len(self.t)],
                                                                                y4[i % len(self.t)]), lw=lineThickness))

        # m1 CW circle
        ax.add_patch(plt.Circle((self.x4, y4[i % len(self.t)]), 0.6, color='k'))

        # Centre Cam Circle
        ax.add_patch(plt.Circle((0, 0), self.L1[i % len(self.t)], color='r'))
        # Draw some arrows on that rotate
        LenLine = 0.9
        LineOffset = 0.0
        ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + LineOffset), LenLine * np.sin(self.z[i, 0] + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + LineOffset), LenLine * np.cos(self.z[i, 0] + LineOffset)), color='k', linestyle='--'))
        ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset)), color='k', linestyle='--'))

        # beam centre circle
        ax.add_patch(plt.Circle((self.xb[i % len(self.t)], self.yb[i % len(self.t)]), 0.05, color='k'))

        # Projectile circle
        ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.5, color='r'))

        #patches.append(ax.text(-3, 3, "Total Energy: %.3f" % totalEnergy[i]))
        #patches.append(ax.text(-3, 1, "T: %.3f" % T[i]))
        #patches.append(ax.text(-3, 2, "V: %.3f" % V[i]))
        #patches.append(ax.text(-3, 2, "Range: %.3f" % Sx[i]))
        # patches.append(ax.add_patch(plt.text(0, 0, "Circle", size=30)))
        return ax


if __name__ == "__main__":

    zinit = np.array([0.01 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0])
    # a: 0.742940789186 x4: b: 0.242264655896 L2: 1.17932428016 L3: 2.16683212908 Range: 151.892725022
    g = 9.81
    m1 = 44.8
    m3 = 1.0
    L2 = 1.17932428016
    L3 = 2.16683212908
    beamDensity = 1.0
    m_b = beamDensity * L3

    #a = 0.742940789186
    a = 0.7
    print('Correct drop length', 1.0 + np.sin(np.pi/4))
    #a = 0.6
    x4 = a + 0.2
    # Numerically solve for b so that the maximum drop is 1.
    b = fsolve(solveAandB, 0.1, args = (a, x4))[0]
    print('a:', a,'x4:', x4, "Solved b:", b)

    endtime = 1.0
    dt = 0.01
    testMurlin = murlinTreb(name="TestTrebOnWheels", beamDensity=beamDensity, g=9.81)

    range, throwAngle = testMurlin.runTreb(m1=m1, m3=m3, L2=L2, L3=L3, x4=x4, endtime=endtime, dt=dt, zinit=zinit,
                                           a=a, b=b)
    print('Range:', range, 'Throw angle:', np.rad2deg(throwAngle))


    # Animate this shit!

    fig = plt.figure(figsize=(10, 6), dpi=80)
    plt.subplot(1, 1, 1, aspect='equal')

    xlimits = plt.xlim(-4, 4)
    ylimits = plt.ylim(-4, 4)
    ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')

    fps = 24
    anim = animation.FuncAnimation(fig, testMurlin.animate, init_func=testMurlin.init, frames=len(testMurlin.t),
                                   interval=1, blit=True)

    plt.show()
