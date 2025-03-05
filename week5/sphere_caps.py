import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u

import argparse
from math import radians

# created Feb. 21 2025 by AJF
# last editing done: Feb. 22 2025

def ra_cap():
    """ Constructs a vector 4-array of a sphereical cap bound by ra = 5 hours

    Parameters:
    ----------
    None

    Returns:
    ----------
    c1_arr : :class: list
        form (x, y, z, 1-h)) = (x, y, z, 1) for cap bound by RA (great circle); h = 0 above 'equator'
    """
    
    # AJF initialize coordinates ; add 90 degrees to RA coordinate; area is 1 - cos(theta), where theta is 90 - dec
    c1 = SkyCoord(ra = (5*u.hourangle + 90*u.deg), dec = 0*u.deg)
    
    # AJF change to cartesian x y z
    c1.representation_type = coord.CartesianRepresentation
    
    # AJF create list with proper formatting
    c1_arr = list([ c1.x.value, c1.y.value, int(c1.z.value), int(np.round(1-np.cos(radians(90))) ) ])
    print(c1_arr)
    
    return c1_arr



def dec_cap():
    """ Changes a certain sky coordinate into the cartesian coordinate representation

    Parameters:
    ----------
    None

    Returns:
    ----------
    c2_arr : :class: list
        form (x, y, z, 1-sin(dec_angle)) ; size of cap is sin(dec)
        
    """
    
    # AJF intitialize coordinates; for dec-bound, ra = 0 and dec = 90 for all; area is 1-sin(dec)
    c2 = SkyCoord(ra = 0*u.deg, dec = 90*u.deg)
    
    # AJF change to cartesian x y z
    c2.representation_type = coord.CartesianRepresentation
    
    # AJF create list with proper formatting
    c2_arr = list([ int(c2.x.value), int(c2.y.value), int(c2.z.value), 1-np.sin(radians(36)) ])
    print(c2_arr)
    
    return c2_arr



def circ_cap():
    """ Changes a certain sky coordinate into the cartesian coordinate representation

    Parameters:
    ----------
    None

    Returns:
    ----------
    c3_arr : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'
    """
    
    # AJF initialize coordinates; since cap is bound in ra and dec != 0/90, then keep as values; area is 1-cos(theta), where theta is "radius"
    c3 = SkyCoord(ra = (5*u.hourangle), dec = 36*u.deg)
    
    # AJF change to cartesian x y z
    c3.representation_type = coord.CartesianRepresentation
    
    # AJF create list with proper formatting
    c3_arr = list([ c3.x.value, c3.y.value, c3.z.value, 1-np.cos(radians(1)) ])
    print(c3_arr)
    
    return c3_arr



def write(c1, c2, c3):
    """ Changes a certain sky coordinate into the cartesian coordinate representation

    Parameters:
    ----------
    c1_arr : :class: list
        form (x, y, z, 1-h)) = (x, y, z, 1) for cap bound by RA (great circle); h = 0 above 'equator'
        
    c2_arr : :class: list
        form (x, y, z, 1-sin(dec_angle)) ; size of cap is sin(dec)
        
    c3 : :class: list
        form (x, y, z, 1-cos(theta)) for cap bound by ra and dec; theta is 'radius'

    Returns:
    ----------
    None - writes text file with final results in proper format
    
    """
    
    # AJF open a text file in write mode in cwd
    out = open('polygons.txt', 'w')
    
    # AJF write opening lines
    out.write('1 polygons\npolygon 1 ( 3 caps, 1 weight, 0 pixel, 0 str):\n')
    
    # AJF write values of sph. cap lists in order; ensure no spaces at end of line
    for i in range(len(c1)):
        if i<(len(c1)-1):
            out.write(str(c1[i])+' ')
        else:
            out.write(str(c1[i]))
    out.write('\n')
    for i in range(len(c2)):
        if i<(len(c2)-1):
            out.write(str(c2[i])+' ')
        else:
            out.write(str(c2[i]))
    out.write('\n')
    for i in range(len(c3)):
        if i<(len(c3)-1):
            out.write(str(c3[i])+' ')
        else:
            out.write(str(c3[i]))
        
    # AJF close out and save text file 
    out.close()



def main(): # AJF executes this section first (highest 'shell' of code)
    # AJF add description
    parser = argparse.ArgumentParser(description='Find values for specific spherical caps and output them to text file')
    
    # AJF execute main functions
    c1 = ra_cap()
    c2 = dec_cap()
    c3 = circ_cap()
    write(c1, c2, c3)


if __name__=='__main__':
    main() 
