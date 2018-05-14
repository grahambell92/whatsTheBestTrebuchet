# Class for the Murlin Trebuchet.
# Created by Graham Bell 16/06/2016

import numpy as np
from numpy import sin, cos, pi, log, sqrt
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import animation
import trebuchetTools

class simpleMurlinTreb:
    def __init__(self,name='treb1', beamDensity = 1.0, g=9.81):
        self.name = name
        self.beamDensity = beamDensity
        self.g = g

    def returnC1(self, theta0, a, b):
        L1Here = a - b * theta0
        d = L1Here * np.sqrt(L1Here ** 2 + b ** 2) / (2 * b)
        e = 0.5 * b * np.log(np.sqrt(L1Here ** 2 + b ** 2) + L1Here)
        C1 = d + e
        return C1

    def L4Length(self, theta, theta0, a, b):
        C1 = self.returnC1(theta0, a, b)
        L4 = C1 - 0.5*b*np.log(a - b*theta + np.sqrt(b**2 + (a - b*theta)**2)) - (a - b*theta)*np.sqrt(b**2 + (a - b*theta)**2)/(2*b)
        return L4

    def dL4Length(self, theta, theta0, a, b):
        dL4 = -0.5*b*(-b*(a - b*theta)/np.sqrt(b**2 + (a - b*theta)**2) - b)/(a - b*theta + np.sqrt(b**2 + (a - b*theta)**2)) + (a - b*theta)**2/(2*np.sqrt(b**2 + (a - b*theta)**2)) + np.sqrt(b**2 + (a - b*theta)**2)/2
        return dL4

    def ddL4Length(self, theta, theta0, a, b):
        ddL4 = b*(b**2*((a - b*theta)/np.sqrt(b**2 + (a - b*theta)**2) + 1)**2/(a - b*theta + np.sqrt(b**2 + (a - b*theta)**2))**2 + b**2*((a - b*theta)**2/(b**2 + (a - b*theta)**2) - 1)/(np.sqrt(b**2 + (a - b*theta)**2)*(a - b*theta + np.sqrt(b**2 + (a - b*theta)**2))) + (a - b*theta)**3/(b**2 + (a - b*theta)**2)**(3/2) - 3*(a - b*theta)/np.sqrt(b**2 + (a - b*theta)**2))/2
        return ddL4

    def solveAandB(self, b, a):
        # Enter in an a value and b is numerically solved.
        zeroRadAngle = a/b + self.theta0
        net = - self.y4Max + self.L4Length(a=a, b=b, theta=zeroRadAngle, theta0=self.theta0)
        return net

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
        L4 = self.L4Length(z[0], self.theta0, self.a, self.b)
        dL4 = self.dL4Length(z[0], self.theta0, self.a, self.b)
        ddL4 = self.ddL4Length(z[0], self.theta0, self.a, self.b)
        # Reverse the gradients of L4 to simulate mass winding up again.
        if z[0] > self.zeroRadiusAngle:
            dL4 = -dL4
            ddL4 = -ddL4
        dz0 = z[2]
        dz1 = z[3]
        dz2 = 3*(L2**2*m3*np.sin(2*z[1] - 2*z[0])*z[2]**2 + 2*L2*L3*m3*np.cos(z[1] - z[0])*z[3]**2 + L2*g*m3*np.sin(2*z[1] - z[0]) + L2*g*m3*np.sin(z[0]) + L2*g*m_b*np.sin(z[0]) - 2*g*m1*dL4 + 2*m1*z[2]**2*ddL4*dL4)/(2*(-3*L2**2*m3*np.cos(2*z[1] - 2*z[0])/2 - 3*L2**2*m3/2 - L2**2*m_b - 3*m1*dL4**2))

        dz3 = (3*L2*(2*L2*L3*m3*np.cos(z[1] - z[0])*z[3]**2 + 2*L2*g*m3*np.sin(z[0]) + L2*g*m_b*np.sin(z[0]) - 2*g*m1*dL4 + 2*m1*z[2]**2*ddL4*dL4)*np.sin(z[1] - z[0]) + 2*(L2*np.cos(z[1] - z[0])*z[2]**2 + g*np.cos(z[1]))*(3*L2**2*m3 + L2**2*m_b + 3*m1*dL4**2))/(2*L3*(3*L2**2*m3*np.cos(z[1] - z[0])**2 + L2**2*m_b + 3*m1*dL4**2))

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

    def runTreb(self, m1=100, m3=1.0,L2=2.0, L3=2.0, x4=0.8+0.2, a=0.8, endtime=2.0, dt=0.01, theta0=np.pi/4,
                y4Max=1.0+np.sin(np.pi/4),
                zinit=np.array([0.01*np.pi/180, 0.1*np.pi/180, 0.0, 0.0])):

        self.zinit = zinit
        self.m1 = m1
        self.m3 = m3

        self.theta0 = self.zinit[0]
        self.y4Max = y4Max
        self.a = a

        self.b = fsolve(self.solveAandB, 0.4, args=(a))[0]

        self.zeroRadiusAngle = self.a / self.b + self.theta0
        #print('a and b', self.a, self.b)

        self.L2 = L2
        self.L3 = L3
        self.h = -np.cos(self.zinit[0]) * self.L2
        self.x4 = x4
        self.m_b = self.beamDensity * (self.L3)

        self.endtime = endtime
        self.dt = dt

        self.t = np.linspace(0, self.endtime, self.endtime / self.dt)


        self.z = odeint(self.deriv, self.zinit, self.t)

        self.L1 = self.a - self.b * (self.z[:, 0])

        self.xb = -self.L2 / 2 * np.sin(self.z[:, 0])
        self.yb = -self.L2 / 2 * np.cos(self.z[:, 0])

        self.x2 = -self.L2 * np.sin(self.z[:, 0])
        self.y2 = -self.L2 * np.cos(self.z[:, 0])

        self.x3 = self.x2 + self.L3 * np.cos(self.z[:, 1])
        self.y3 = self.y2 - self.L3 * np.sin(self.z[:, 1])

        # L4 and y4 are for plotting
        self.L4 = self.L4Length(theta=self.z[:, 0], theta0=self.theta0, a=self.a, b=self.b)

        self.y4 = -self.L4
        for thetaIndex, theta in enumerate(self.z[:, 0]):
            if theta > self.zeroRadiusAngle:
                self.y4[thetaIndex] = -self.y4Max + (self.L4[thetaIndex]-self.y4Max)

        # Grab the largest range value. ThrowRange function args: theta,thetadot,alpha,alphadot
        self.Sx, self.x3Dot, self.y3Dot, self.maxThrowIndex, self.maxThrowRange, self.SxMaxThrowAngle = \
            trebuchetTools.throwRange(xPos=self.x3, yPos=self.y3, dt=self.dt, alpha=self.z[:, 1], g=self.g, h=self.h,
                                 validateThrow=True, returnMaxes=True)

        return self.maxThrowRange, self.SxMaxThrowAngle

    def totalKineticEnergy(self):
        theta = self.z[:, 0]
        dL4 = self.dL4Length(theta, self.theta0, self.a, self.b)
        L4 = self.L4Length(theta, self.theta0, self.a, self.b)
        TKE = self.L2**2*self.m_b*self.z[:, 2]**2/6 + self.m1*self.z[:, 2]**2*dL4**2/2 + self.m3*((self.L2*np.sin(self.z[:, 0])*self.z[:, 2] - self.L3*np.cos(self.z[:, 1])*self.z[:, 3])**2 + (-self.L2*np.cos(self.z[:, 0])*self.z[:, 2] - self.L3*np.sin(self.z[:, 1])*self.z[:, 3])**2)/2
        return TKE

    def totalPotentialEnergy(self):

        TPE = self.m3 * self.g * self.y3 + self.m1 * self.g * (self.y4 + self.y4Max) + self.m_b * self.g * self.yb
        return TPE

    def init(self):
        return []

    def animate(self, i):
        patches = []

        #L4 = self.L4Length(theta=self.z[i, 0], theta0=self.theta0, a=self.a, b=self.b)

        #y4 = -L4

        #if self.z[i, 0] > self.zeroRadiusAngle:
        #    y4 = -self.y4Max + (L4-self.y4Max)
        xL1 = self.L1[i]
        yL1 = 0.0

        # Vertical and horizontal lines:
        # Horizontal rail
        patches.append(ax.add_line(plt.Line2D((-1, 1), (0, 0), lw=1.7, alpha=0.3, color='k', linestyle='--')))
        # Vertical Rail
        patches.append(ax.add_line(plt.Line2D((0, 0), (-1.5, 1.5), lw=1.7, alpha=0.3, color='k', linestyle='--')))

        # 0, 0 - x2,y2 line
        patches.append(ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=2.5)))

        # x2,y2-x3,y3 line
        patches.append(ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=2.5)))

        # Draw string
        # End 1 is Cam, end 2 is CW
        patches.append(ax.add_line(plt.Line2D((xL1, xL1), (yL1, self.y4[i % len(self.t)]), lw=1.8)))

        # m1 CW circle
        patches.append(ax.add_patch(plt.Circle((xL1, self.y4[i % len(self.t)]), 0.2, color='k')))

        # Centre Cam Circle
        patches.append(ax.add_patch(plt.Circle((0, 0), xL1, color='r', alpha=0.5)))
        # Draw some arrows on that rotate
        LenLine = 0.6
        LineOffset = 0.0
        patches.append(ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + LineOffset), LenLine * np.sin(self.z[i, 0] + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + LineOffset), LenLine * np.cos(self.z[i, 0] + LineOffset)), color='k', linestyle='--')))
        patches.append(ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset)), (-LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset), LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset)), color='k', linestyle='--')))

        # beam centre circle
        patches.append(ax.add_patch(plt.Circle((self.xb[i % len(self.t)], self.yb[i % len(self.t)]), 0.05, color='k')))

        # Projectile circle
        patches.append(ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), 0.1, color='r')))

        return patches

    def plotTreb(self, ax, i, lineThickness=0.4, railThickness=0.3, CWRadius=0.2, projRadius=0.5, beamCentreRadius=0.2,
                 imageAlpha=1.0):
        # i is timestep.

        # Max length of L4 before bouncing up.

        xL1 = self.L1[i]
        yL1 = 0.0

        # Vertical and horizontal lines:
        # Horizontal rail
        dashPattern = [3, 2]
        ax.add_line(plt.Line2D((-2, 2), (0, 0), lw=railThickness, alpha=0.3*imageAlpha, color='0.2', dashes=dashPattern,
                               zorder=0))
        # Vertical Rail
        ax.add_line(plt.Line2D((0, 0), (-2, 2), lw=railThickness, alpha=0.3*imageAlpha, color='0.2', dashes=dashPattern,
                               zorder=0))


        # 0, 0 - x2,y2 line
        ax.add_line(plt.Line2D((0, self.x2[i % len(self.t)]), (0, self.y2[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, color='k', zorder=1))

        # x2,y2-x3,y3 line
        ax.add_line(plt.Line2D((self.x2[i % len(self.t)], self.x3[i % len(self.t)]),
                                              (self.y2[i % len(self.t)], self.y3[i % len(self.t)]), lw=lineThickness,
                               alpha=imageAlpha, color='k', zorder=1))

        # Draw string
        # End 1 is Cam, end 2 is CW
        ax.add_line(plt.Line2D((xL1, xL1), (yL1, self.y4[i % len(self.t)]), color='k',
                               lw=lineThickness, alpha=imageAlpha, zorder=1))



        # Centre Cam Circle
        ax.add_patch(plt.Circle((0, 0), xL1, color='#69d0e7', alpha=0.5*imageAlpha, zorder=1))
        # Draw some arrows on that rotate
        LenLine = 0.6
        LineOffset = 0.0
        ax.add_line(
            plt.Line2D((-LenLine * np.sin(self.z[i, 0] + LineOffset), LenLine * np.sin(self.z[i, 0] + LineOffset)),
                       (-LenLine * np.cos(self.z[i, 0] + LineOffset), LenLine * np.cos(self.z[i, 0] + LineOffset)),
                       color='k', linestyle='--', alpha=imageAlpha, zorder=1))
        ax.add_line(plt.Line2D((-LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset),
                                               LenLine * np.sin(self.z[i, 0] + np.pi / 2 + LineOffset)), (
                                              -LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset),
                                              LenLine * np.cos(self.z[i, 0] + np.pi / 2 + LineOffset)), color='k',
                                              linestyle='--', alpha=imageAlpha, zorder=1))

        # beam centre circle
        ax.add_patch(plt.Circle((self.xb[i % len(self.t)], self.yb[i % len(self.t)]), beamCentreRadius, color='k',
                     alpha=imageAlpha, zorder=2))

        # Projectile circle
        ax.add_patch(plt.Circle((self.x3[i % len(self.t)], self.y3[i % len(self.t)]), projRadius, color='#f38630',
                                alpha=imageAlpha, zorder=2))

        # m1 CW circle
        ax.add_patch(plt.Circle((xL1, self.y4[i % len(self.t)]), CWRadius, color='#e0e4cc', alpha=imageAlpha, zorder=2))
        return ax

    def optimiserThrowTreb(self, x, returnThrowAngle=False):
        [L2, L3, a] = x
        throwRange, throwAngle = self.runTreb(m1=self.m1, m3=self.m3, L2=L2, L3=L3, endtime=self.endtime, dt=self.dt,
                                                  zinit=self.zinit, a=a)
        b = self.b
        #print("Optimizer Trying:", 'a:', self.a, 'b:', b, 'L2:', L2, 'L3:', L3, "Range:", throwRange, 'throw angle:',
        #      np.rad2deg(throwAngle))
        if np.isnan(throwRange):
            throwRange = 0.01
        if returnThrowAngle is True:
            return -throwRange, throwAngle
        else:
            return -throwRange


if __name__ == "__main__":

    zinit = np.array([45 * np.pi / 180, 0.1 * np.pi / 180, 0.0, 0.0])
    # Optimised result

    g = 9.81
    m1 = 100.937
    m3 = 1.0
    L2 = 3.51492
    L3 = 4.81832
    beamDensity = 1.0
    m_b = beamDensity * L3

    endtime = 1.5
    dt = 0.01
    treb = simpleMurlinTreb(name="Simple Murlin Treb", beamDensity=beamDensity, g=9.81)

    a = 1.22

    y4Max = 1.0 + np.sin(np.pi/4)
    print('Correct drop length', y4Max)


    range, throwAngle = treb.runTreb(m1=m1, m3=m3, L2=L2, L3=L3, endtime=endtime, dt=dt, zinit=zinit,
                                           a=a, y4Max=y4Max)

    print('Range:', range, 'Throw angle:', np.rad2deg(throwAngle))

    if True:
        CWEnergy = 9.81 * 100 * (y4Max)
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

    # Animate this shit!

    fig = plt.figure(figsize=(10, 6), dpi=80)
    plt.subplot(1, 1, 1, aspect='equal')

    bounds = 6
    xlimits = plt.xlim(-bounds, bounds)
    ylimits = plt.ylim(-bounds, bounds)
    ax = plt.axes(xlim=xlimits, ylim=ylimits, aspect='equal')

    fps = 24
    anim = animation.FuncAnimation(fig, treb.animate, init_func=treb.init, frames=len(treb.t),
                                   interval=1, blit=True)

    plt.show()
