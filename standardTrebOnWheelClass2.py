# Created by Graham Bell 08/07/2016
# Model for estimating standard trebuchet on wheels motion.

import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import trebuchetTools

class standardTrebOnWheels:
    def __init__(self, name='treb1', beamDensity=1.0, g=9.81):
        self.name = name
        self.beamDensity = beamDensity
        self.g = g

    def deriv(self, z, t):
        L1 = self.L1
        L2 = self.L2
        L3 = self.L3
        L4 = self.L4
        Lb = self.Lb
        g = self.g
        m1 = self.m1
        m3 = self.m3
        m_b = self.m_b
        m_c = self.m_c


        # z[0] = theta
        # z[1] = phi
        # z[2] = alpha
        # z[3] = x0
        # z[4] = theta_dot
        # z[5] = phi_dot
        # z[6] = alpha_dot
        # z[7] = x0_dot

        dz0 = z[4]
        dz1 = z[5]
        dz2 = z[6]
        dz3 = z[7]
        dz4 = 3*(-2*m1*(L1*np.sin(z[1] + z[0])*z[4]**2 - g*np.sin(z[1]))*(L1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0]) - L1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0]) + L2*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1]) + (L1*m1 - L2*m3 - Lb*m_b - Lb*m_c)*np.cos(z[1])*np.cos(z[0])) - 2*m3*(L2*np.cos(z[2] - z[0])*z[4]**2 + g*np.cos(z[2]))*(L1*m1*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + L2*m1*np.sin(z[2] - z[0])*np.cos(z[1])**2 - L2*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0]) + (-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2])*np.cos(z[0])) - 2*(L1*m1*np.cos(z[1] + z[0])*np.cos(z[1]) - L2*m3*np.sin(z[2] - z[0])*np.sin(z[2]) + (-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[0]))*(L1*m1*np.sin(z[0])*z[4]**2 - L2*m3*np.sin(z[0])*z[4]**2 + L3*m3*np.cos(z[2])*z[6]**2 + L4*m1*np.sin(z[1])*z[5]**2 - Lb*m_b*np.sin(z[0])*z[4]**2 - Lb*m_c*np.sin(z[0])*z[4]**2) + (-m1*np.sin(z[1])**2 + m3*np.sin(z[2])**2 - m3 - m_b - m_c)*(2*L1*L4*m1*np.sin(z[1] + z[0])*z[5]**2 + 2*L1*g*m1*np.sin(z[0]) - 2*L2*L3*m3*np.cos(z[2] - z[0])*z[6]**2 - 2*L2*g*m3*np.sin(z[0]) + Lb**2*m_b*np.sin(2*z[0])*z[4]**2 + Lb**2*m_c*np.sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*np.sin(z[0])))/(2*(-3*L1**2*m1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0])**2 - 6*L1*L2*m1*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[1])*np.cos(z[0]) - 3*L2**2*m1*m3*np.sin(z[2] - z[0])**2*np.cos(z[1])**2 + 3*L2**2*m3*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2))
        dz5 = (-2*m3*(L2*np.cos(z[2] - z[0])*z[4]**2 + g*np.cos(z[2]))*(3*L1*L2*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])*np.cos(z[1] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[0]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.cos(z[1])*np.cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])*np.cos(z[1])) + 2*(L1*np.sin(z[1] + z[0])*z[4]**2 - g*np.sin(z[1]))*(3*L2**2*m3*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[0]) + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2) - 3*(L1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0]) - L1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0]) + L2*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1]) + (L1*m1 - L2*m3 - Lb*m_b - Lb*m_c)*np.cos(z[1])*np.cos(z[0]))*(2*L1*L4*m1*np.sin(z[1] + z[0])*z[5]**2 + 2*L1*g*m1*np.sin(z[0]) - 2*L2*L3*m3*np.cos(z[2] - z[0])*z[6]**2 - 2*L2*g*m3*np.sin(z[0]) + Lb**2*m_b*np.sin(2*z[0])*z[4]**2 + Lb**2*m_c*np.sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*np.sin(z[0])) - 2*(3*L1*L2*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[0]) + 3*L2**2*m3*np.sin(z[2] - z[0])**2*np.cos(z[1]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1]))*(L1*m1*np.sin(z[0])*z[4]**2 - L2*m3*np.sin(z[0])*z[4]**2 + L3*m3*np.cos(z[2])*z[6]**2 + L4*m1*np.sin(z[1])*z[5]**2 - Lb*m_b*np.sin(z[0])*z[4]**2 - Lb*m_c*np.sin(z[0])*z[4]**2))/(2*L4*(-3*L1**2*m1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0])**2 - 6*L1*L2*m1*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[1])*np.cos(z[0]) - 3*L2**2*m1*m3*np.sin(z[2] - z[0])**2*np.cos(z[1])**2 + 3*L2**2*m3*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2))
        dz6 = (-2*m1*(L1*np.sin(z[1] + z[0])*z[4]**2 - g*np.sin(z[1]))*(3*L1*L2*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])*np.cos(z[1] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[0]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.cos(z[1])*np.cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])*np.cos(z[1])) + 2*(L2*np.cos(z[2] - z[0])*z[4]**2 + g*np.cos(z[2]))*(3*L1**2*m1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0])**2 + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[1])*np.cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2) + 2*(3*L1**2*m1*np.sin(z[2])*np.cos(z[1] + z[0])**2 + 3*L1*L2*m1*np.sin(z[2] - z[0])*np.cos(z[1] + z[0])*np.cos(z[1]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2]))*(L1*m1*np.sin(z[0])*z[4]**2 - L2*m3*np.sin(z[0])*z[4]**2 + L3*m3*np.cos(z[2])*z[6]**2 + L4*m1*np.sin(z[1])*z[5]**2 - Lb*m_b*np.sin(z[0])*z[4]**2 - Lb*m_c*np.sin(z[0])*z[4]**2) - 3*(L1*m1*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + L2*m1*np.sin(z[2] - z[0])*np.cos(z[1])**2 - L2*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0]) + (-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2])*np.cos(z[0]))*(2*L1*L4*m1*np.sin(z[1] + z[0])*z[5]**2 + 2*L1*g*m1*np.sin(z[0]) - 2*L2*L3*m3*np.cos(z[2] - z[0])*z[6]**2 - 2*L2*g*m3*np.sin(z[0]) + Lb**2*m_b*np.sin(2*z[0])*z[4]**2 + Lb**2*m_c*np.sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*np.sin(z[0])))/(2*L3*(-3*L1**2*m1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0])**2 - 6*L1*L2*m1*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[1])*np.cos(z[0]) - 3*L2**2*m1*m3*np.sin(z[2] - z[0])**2*np.cos(z[1])**2 + 3*L2**2*m3*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2))
        dz7 = (-2*m1*(L1*np.sin(z[1] + z[0])*z[4]**2 - g*np.sin(z[1]))*(3*L1*L2*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0]) - 3*L1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[0]) + 3*L2**2*m3*np.sin(z[2] - z[0])**2*np.cos(z[1]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])) + 2*m3*(L2*np.cos(z[2] - z[0])*z[4]**2 + g*np.cos(z[2]))*(3*L1**2*m1*np.sin(z[2])*np.cos(z[1] + z[0])**2 + 3*L1*L2*m1*np.sin(z[2] - z[0])*np.cos(z[1] + z[0])*np.cos(z[1]) + 3*L2*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.cos(z[0]) - (3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])) - 3*(L1*m1*np.cos(z[1] + z[0])*np.cos(z[1]) - L2*m3*np.sin(z[2] - z[0])*np.sin(z[2]) + (-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[0]))*(2*L1*L4*m1*np.sin(z[1] + z[0])*z[5]**2 + 2*L1*g*m1*np.sin(z[0]) - 2*L2*L3*m3*np.cos(z[2] - z[0])*z[6]**2 - 2*L2*g*m3*np.sin(z[0]) + Lb**2*m_b*np.sin(2*z[0])*z[4]**2 + Lb**2*m_c*np.sin(2*z[0])*z[4]**2 - 2*Lb*g*m_b*np.sin(z[0])) + (L1*m1*np.sin(z[0])*z[4]**2 - L2*m3*np.sin(z[0])*z[4]**2 + L3*m3*np.cos(z[2])*z[6]**2 + L4*m1*np.sin(z[1])*z[5]**2 - Lb*m_b*np.sin(z[0])*z[4]**2 - Lb*m_c*np.sin(z[0])*z[4]**2)*(6*L1**2*m1*np.cos(z[1] + z[0])**2 - 6*L1**2*m1 - 2*L1**2*m_b + 2*L1*L2*m_b + 6*L2**2*m3*np.sin(z[2] - z[0])**2 - 6*L2**2*m3 - 2*L2**2*m_b - 3*Lb**2*m_b*np.cos(2*z[0]) - 3*Lb**2*m_b - 3*Lb**2*m_c*np.cos(2*z[0]) - 3*Lb**2*m_c))/(2*(-3*L1**2*m1*m3*np.sin(z[2])**2*np.cos(z[1] + z[0])**2 + 3*L1**2*m1*(m1 + m3 + m_b + m_c)*np.cos(z[1] + z[0])**2 - 6*L1*L2*m1*m3*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[1] + z[0])*np.cos(z[1]) + 6*L1*m1*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.cos(z[1] + z[0])*np.cos(z[1])*np.cos(z[0]) - 3*L2**2*m1*m3*np.sin(z[2] - z[0])**2*np.cos(z[1])**2 + 3*L2**2*m3*(m1 + m3 + m_b + m_c)*np.sin(z[2] - z[0])**2 - 6*L2*m3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)*np.sin(z[2] - z[0])*np.sin(z[2])*np.cos(z[0]) + m1*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.cos(z[1])**2 + m3*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2)*np.sin(z[2])**2 - (m1 + m3 + m_b + m_c)*(3*L1**2*m1 + L1**2*m_b - L1*L2*m_b + 3*L2**2*m3 + L2**2*m_b + 3*Lb**2*m_b*np.cos(z[0])**2 + 3*Lb**2*m_c*np.cos(z[0])**2) + 3*(-L1*m1 + L2*m3 + Lb*m_b + Lb*m_c)**2*np.cos(z[0])**2))
        return np.array([dz0, dz1, dz2, dz3, dz4, dz5, dz6, dz7])

    def runTreb(self, m1=40.0, L2=1.0, L3=1.0, m3=1.0, L1=1.0, endtime=1.5, dt=0.005, zinit=np.array([45*pi/180, 1*pi/180, 1*pi/180, 0.1, 0, 0, 0, 0])):
        self.m1 = m1
        self.m3 = m3
        self.m_c = 1.0#0.1*m1
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.m_b = (self.L1 + self.L2) * self.beamDensity
        self.h = -np.cos(45 * np.pi / 180) * self.L2
        self.L4 = - self.L1 - self.h
        if self.L4 < 0.1:
            self.L4 = 0.1

        self.zinit = zinit
        self.Lb = (self.L2 - self.L1) / 2.0
        self.m_b = (self.L1 + self.L2) * self.beamDensity
        self.h = -np.cos(self.zinit[0]) * self.L2

        self.endtime = endtime
        self.dt = dt
        self.t = np.linspace(0, endtime, endtime/dt)
        self.z = odeint(self.deriv, self.zinit, self.t)
        z = self.z

        theta = self.z[:, 0]
        phi = self.z[:, 1]
        alpha = self.z[:, 2]
        self.x0 = self.z[:, 3]
        self.x1 = self.L1 * np.sin(theta) + self.x0
        self.y1 = self.L1 * np.cos(theta)
        self.x2 = -self.L2 * np.sin(theta) + self.x0
        self.y2 = -self.L2 * np.cos(theta)

        self.x3 = self.x2 + self.L3 * np.cos(alpha)
        self.y3 = self.y2 - self.L3 * np.sin(alpha)

        self.x4 = self.x1 + self.L4 * np.sin(phi)
        self.y4 = self.y1 - self.L4 * np.cos(phi)
        self.yb = -self.Lb * np.cos(theta)
        self.xb = -self.Lb * np.sin(theta) + self.x0

        # z[0] = theta
        # z[1] = phi
        # z[2] = alpha
        # z[3] = x0
        # z[4] = theta_dot
        # z[5] = phi_dot
        # z[6] = alpha_dot
        # z[7] = x0_dot
        # ThrowRange(theta,thetadot,alpha,alphadot,x0, x0_dot):

        # Grab the largest range value.
        # Grab the largest range value.
        self.Sx, self.x3Dot, self.y3Dot, self.maxThrowIndex, self.maxThrowRange, self.SxMaxThrowAngle = \
            trebuchetTools.throwRange(xPos=self.x3, yPos=self.y3, dt=self.dt, alpha=self.z[:, 2], g=self.g, h=self.h,
                                 validateThrow=True, returnMaxes=True)
        return self.maxThrowRange, self.SxMaxThrowAngle


    def totalKineticEnergy(self):
        TKE = self.Lb**2*(self.m_b + self.m_c)*np.cos(self.z[:, 0])**2*self.z[:, 4]**2/2 + self.m1*((-self.L1*np.sin(self.z[:, 0])*self.z[:, 4] + self.L4*np.sin(self.z[:, 1])*self.z[:, 5])**2 + (self.L1*np.cos(self.z[:, 0])*self.z[:, 4] + self.L4*np.cos(self.z[:, 1])*self.z[:, 5] + self.z[:, 7])**2)/2 + self.m3*((self.L2*np.sin(self.z[:, 0])*self.z[:, 4] - self.L3*np.cos(self.z[:, 2])*self.z[:, 6])**2 + (-self.L2*np.cos(self.z[:, 0])*self.z[:, 4] - self.L3*np.sin(self.z[:, 2])*self.z[:, 6] + self.z[:, 7])**2)/2 + self.m_b*(self.L1**2 - self.L1*self.L2 + self.L2**2)*self.z[:, 4]**2/6
        return TKE

    def totalPotentialEnergy(self):
        CWRefZeroHeight = self.L1 + self.L4
        TPE = -self.Lb*self.g*self.m_b*np.cos(self.z[:, 0]) + self.g*self.m1*(CWRefZeroHeight + self.L1*np.cos(self.z[:, 0]) - self.L4*np.cos(self.z[:, 1])) + self.g*self.m3*(-self.L2*np.cos(self.z[:, 0]) - self.L3*np.sin(self.z[:, 2]))
        return TPE

    # Present an animation
    def init(self):
        return []

    def animate(self, i):
        patches = []
        trebarmwidth = 1.0
        # L1 line
        patches.append(ax.add_line(plt.Line2D((self.x0[i % len(self.t)], self.x1[i % len(self.t)]),
                                              (0, self.y1[i % len(self.t)]), lw=trebarmwidth)))

        # L2 line
        patches.append(ax.add_line(plt.Line2D((self.x0[i % len(self.t)], self.x2[i % len(self.t)]),
                                              (0, self.y2[i % len(self.t)]), lw=trebarmwidth)))
        # L4 line
        patches.append(ax.add_line(
            plt.Line2D((self.x1[i % len(self.t)], self.x4[i % len(self.t)]), (self.y1[i % len(self.t)],
                                                                              self.y4[i % len(self.t)]), lw=trebarmwidth)))
        # L3 line
        patches.append(ax.add_line(
            plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]), (self.y2[i % len(self.t)],
                                                        self.y3[i % len(self.t)]), lw=trebarmwidth, color='c')))

        # x1 hinge position circle
        # patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
        # CW circle
        patches.append(ax.add_patch(plt.Circle((self.x4[i % len(self.t)], self.y4[i % len(self.t)]), 0.2, color='k')))
        # Beam tip circle
        # patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))
        # Projectile tip circle
        patches.append(ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.1, color='r')))
        # Center dot
        patches.append(ax.add_patch(plt.Circle((self.x0[i % len(self.t)], 0), 0.05, color='k')))

        return patches

    def plotTreb(self, ax, i, lineThickness=0.4, railThickness=0.3, CWRadius=0.2, projRadius=0.5, beamCentreRadius=0.2,
                 imageAlpha=1.0):
        # L1 line
        ax.add_line(plt.Line2D((self.x0[i % len(self.t)], self.x1[i % len(self.t)]),
                                              (0, self.y1[i % len(self.t)]), lw=lineThickness, alpha=imageAlpha,
                               zorder=2, color='k'))

        # L2 line
        ax.add_line(plt.Line2D((self.x0[i % len(self.t)], self.x2[i % len(self.t)]),
                                              (0, self.y2[i % len(self.t)]), lw=lineThickness, alpha=imageAlpha,
                               zorder=2, color='k'))
        # L4 line
        ax.add_line(plt.Line2D((self.x1[i % len(self.t)],
                                self.x4[i % len(self.t)]), (self.y1[i % len(self.t)],
                                                            self.y4[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, zorder=2, color='k'))
        # L3 line
        ax.add_line(
            plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]), (self.y2[i % len(self.t)],
                                                        self.y3[i % len(self.t)]), lw=lineThickness,
                       alpha=imageAlpha, zorder=2, color='k'))

        # x1 hinge position circle
        # patches.append(ax.add_patch(plt.Circle((x1[i % len(t)],y1[i % len(t)]),0.1,color='b') ))
        # CW circle
        ax.add_patch(plt.Circle((self.x4[i % len(self.t)], self.y4[i % len(self.t)]), CWRadius, color='#69d0e7',
                                alpha=imageAlpha, zorder=3))
        # Beam tip circle
        # patches.append(ax.add_patch(plt.Circle((x2[i % len(t)],y2[i % len(t)]),0.1,color='r') ))
        # Projectile tip circle
        ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), projRadius, color='#f38630',
                                alpha=imageAlpha, zorder=3))
        # Center dot
        ax.add_patch(plt.Circle((self.x0[i % len(self.t)], 0), beamCentreRadius, color='#e0e4cc', alpha=imageAlpha,
                                zorder=3))
        return ax

    def optimiserThrowTreb(self, x, returnThrowAngle=False):
        [L2, L3] = x
        throwRange, throwAngle = self.runTreb(m1=self.m1, m3=self.m3, L2=L2, L3=L3, endtime=self.endtime,
                                                  dt=self.dt, zinit=self.zinit)
        if np.isnan(throwRange):
            throwRange = 0.1
        print("Optimizer Trying:", 'L2:', L2, 'L3:', L3, 'Range:', throwRange, 'throw angle:', np.rad2deg(throwAngle))
        if returnThrowAngle is True:
            return -throwRange, throwAngle
        else:
            return -throwRange


if __name__ == "__main__":
    treb = standardTrebOnWheels(name="TestTrebOnWheels")
    zinit = np.array([45 * pi / 180, 1 * pi / 180, 1 * pi / 180, 0.1, 0, 0, 0, 0])
    range, throwAngle = treb.runTreb(m1=100.3, zinit=zinit, L2=0.683, L3=1.624, endtime=1.5)
    print('Range', range, 'throw angle', np.rad2deg(throwAngle))

    if True:
        CWEnergy = 9.81 * 100 * (1.0 + np.sin(np.pi / 4))
        print(CWEnergy)
        TKE = treb.totalKineticEnergy()
        TPE = treb.totalPotentialEnergy()
        projPE = trebuchetTools.projPE(mass=treb.m3, yPos=treb.y3, g=treb.g, refHeight=treb.h)
        projKE = trebuchetTools.projKE(mass=treb.m3, xDot=treb.x3Dot, yDot=treb.y3Dot)
        plt.plot(treb.t, TKE, label='TKE')
        plt.plot(treb.t, TPE, label='TPE')
        plt.plot(treb.t, projKE, label='projKE')
        plt.plot(treb.t, projPE, label='projPE')
        plt.axhline(y=CWEnergy)
        plt.axvline(x=treb.t[treb.maxThrowIndex])
        plt.legend()
        plt.show()
        CWEnergy = 9.81*100*(1.0+np.sin(np.pi/4))
        print(CWEnergy)

    # Animate

    import matplotlib.pyplot as plt
    from matplotlib import animation
    
    fig = plt.figure(figsize=(10, 6), dpi=80)
    plt.subplot(1, 1, 1, aspect='equal')
    
    xlimits = plt.xlim(-5, 5)
    ylimits = plt.ylim(-5, 5)
    ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')
    
    fps = 24
    anim = animation.FuncAnimation(fig, treb.animate,
                    init_func=treb.init, frames=len(treb.t), interval=1, blit=True)
    plt.show()