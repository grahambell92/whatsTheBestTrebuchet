# Script with common tools used in the Trebuchet models for consistency.
# Written by Graham Bell 06/04/2016
import numpy as np

def throwRange(xPos, yPos, dt, alpha=None, g=9.81, h=0.0, validateThrow=False, returnMaxes=False):
    # Just take the numerical gradient of the x and y positions for easy methods to get velocities.
    # Alpha is the angle of the string to horizontal.
    x3Dot = np.gradient(xPos, edge_order=2) / dt
    y3Dot = np.gradient(yPos, edge_order=2) / dt
    Sx = x3Dot * ((y3Dot + np.sqrt((y3Dot ** 2) + (2 * g * (yPos - h)))) / g) + xPos
    if validateThrow is True:
        Sx[x3Dot < 0] = np.nan
        Sx[y3Dot < 0] = np.nan
        Sx[alpha > 2 * np.pi] = np.nan
    if returnMaxes is True:
        try:
            maxThrowIndex = np.nanargmax(Sx)
            maxThrowRange = Sx[maxThrowIndex]
            SxMaxThrowAngle = np.arctan2(y3Dot[maxThrowIndex], x3Dot[maxThrowIndex])
        except:
            SxMaxThrowAngle = np.nan
            maxThrowRange = np.nan
            maxThrowIndex = 0
        return Sx, x3Dot, y3Dot, maxThrowIndex, maxThrowRange, SxMaxThrowAngle
    return Sx, x3Dot, y3Dot

def projPE(mass, yPos, g=9.81, refHeight=0.0):
    PEproj = mass * g * (yPos + np.abs(refHeight))
    return PEproj

def projKE(mass, xDot, yDot):
    VelMag = np.sqrt(xDot**2 + yDot**2)
    KEproj = 0.5*mass*(VelMag**2)
    return KEproj

if __name__ == "__main__":
    pass

